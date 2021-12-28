#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {
    // input: input tensor value (it sores the id)
    // cols: How many elements to do SparseSegmentSum
    // output: rows * embedding_size
    template<typename T>
    static void sparse_gather_v1(T *input, int rows, int cols, 
                                float *embedding_table, float *output,
                                int embedding_size, bool is_mean) {
    T *pidx = input;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < embedding_size; ++j) {
        float value = 0;
        int dense_num = 0;
        for (int k = 0; k < cols; ++k) {
            int embedding_row = (int)pidx[k];
            if (embedding_row >= 0) {
            value += embedding_table[embedding_row * embedding_size + j];
            dense_num += 1;
            }
        }

        if (is_mean && dense_num > 0) {
            *output++ = value / dense_num;
        } else {
            *output++ = value;
        }
        }
        pidx += cols;
    }
    }

    // embedding_size = 1
    template<typename T>
    static void sparse_gather_embeddingsize1(T *input, int rows, int cols,
                                            float *embedding_table, float *output,
                                            bool is_mean) {
    T *pidx = input;
    for (int i = 0; i < rows; ++i) {
        float value = 0;
        int dense_num = 0;
        for (int k = 0; k < cols; ++k) {
        int embedding_row = pidx[k];
        if (embedding_row >= 0) {
            value += embedding_table[embedding_row];
            dense_num += 1;
        }
        }
        if (is_mean && dense_num > 0) {
        *output++ = value / dense_num;
        } else {
        *output++ = value;
        }
        pidx += cols;
    }
    }

    // input cols = 1
    template<typename T>
    static void sparse_gather_column1(T *input, int rows, float *embedding_table, 
                            float *output, int embedding_size) {
    T *pidx = input;
    for (int i = 0; i < rows; ++i) {
        int embedding_row = *pidx++;
        if (embedding_row >= 0) {
        float *pembedding = &embedding_table[embedding_row * embedding_size];
        for (int j = 0; j < embedding_size; ++j) {
            output[j] = pembedding[j];
        }
        } else {
        for (int j = 0; j < embedding_size; ++j) {
            output[j] = 0;
        }
        }
        output += embedding_size;
    }
    }

    template<typename T>
    static void sparse_gather(T *input, int rows, int cols, float *embedding_table,
                            float *output, int embedding_size, bool is_mean) {
    if (embedding_size == 1) {
        sparse_gather_embeddingsize1(input, rows, cols, embedding_table, output,
                                    is_mean);
    } else if (cols == 1) {
        sparse_gather_column1(input, rows, embedding_table, output, embedding_size);
    } else {
        //printf("General sparse gather!\n");
        sparse_gather_v1(input, rows, cols, embedding_table, output, embedding_size, 
                        is_mean);
    }
    }

    // Use memcpy or manually assign?
    static void mycopy(float *dst, float *src, int float_num) {
    memcpy(dst, src, float_num * sizeof(float));
    }

    static void myadd(float *dst, float *src, int float_num) {
    for (int i = 0; i < float_num; ++i) {
        dst[i] += src[i];
    }
    }

    static void myscale(float *dst, float factor, int float_num) {
    for (int i = 0; i < float_num; ++i) {
        dst[i] *= factor;
    }
    }

    template<typename Tid, typename Tshape>
    static void sparse_gather(Tid *input, int64 input_size, Tshape *indice,
                            int indice_dim, Tshape *shape, int rows, int cols,
                            float *embedding_table, float *output,
                            int embedding_size, bool is_mean) {
    // Record how many values in each row
    int *row_values = new int[rows];
    memset(row_values, 0, rows * sizeof(int));

    for (int64 i = 0; i < input_size; ++i) {
        Tid id = input[i];
        if (id < 0) { // Skip invalid id
        continue;
        }
        auto row = indice[i * indice_dim];
        for (int k = 1; k < indice_dim - 1; ++k) {
        row = row * shape[k] + indice[i * indice_dim + k];
        }
        if (row_values[row] > 0) {
        myadd(&output[row * embedding_size], 
                &embedding_table[id * embedding_size], embedding_size);
        } else {
        mycopy(&output[row * embedding_size],
                &embedding_table[id * embedding_size], embedding_size);
        }
        row_values[row] += 1;
    }

    for (int i = 0; i < rows; ++i) {
        if (row_values[i] == 0) {
        memset(&output[i * embedding_size], 0, embedding_size * sizeof(float));
        } else if (is_mean && row_values[i] > 1) {
        float factor = 1.0f / row_values[i];
        myscale(&output[i * embedding_size], factor, embedding_size);
        }
    }

    delete[] row_values;
    }
}

/*
  sample: [['green' 'red' 'blue' 'yellow' 'pink' 'blue' 'red' 'indigo']
           ['' '' '' '' '' '' '' '']
           ['' '' '' 'yellow' 'pink' 'blue' 'red' 'indigo']
           ['' '' '' '' '' '' '' '']
           ['green' '' '' '' '' '' '' '']]
     =>   [[ True  True  True  True  True  True  True  True]
           [False False False False False False False False]
           [False False False  True  True  True  True  True]
           [False False False False False False False False]
           [ True False False False False False False False]]
--------------------------------------------------------------------------------------
  weight: float[[ 0.23860918  0.07992432 -0.7441818 ]
                [-0.8256738  -0.50271106  0.39016065]
                [-0.7978571   0.3993331  -0.12494776]
                [-0.555991   -0.6705441  -0.23192379]
                [-0.5283828   0.19715567  0.12184268]]
  input: int64[4 0 0 1 1 0 0 1 1 1 0 0 1 4] from StringToHashBucketFast output
  dense_shape: int64[5 8]
  indice: int64[[0 0] from to_sparse_input/indices(Where) output
                [0 1]
                [0 2]
                [0 3]
                [0 4]
                [0 5]
                [0 6]
                [0 7]
                [2 3]
                [2 4]
                [2 5]
                [2 6]
                [2 7]
                [4 0]]
    embedded: float[[-0.25637093 -0.12391002 -0.21055032]
                    [ 0.          0.          0.        ]
                    [-0.3999606  -0.2696569  -0.06357633]
                    [ 0.          0.          0.        ]
                    [-0.5283828   0.19715567  0.12184268]]
-----------------------------------------------------------------------------------
      input_size: sum of input tensor size == 14
      indice_dim: dim_size(1) of indice tensor[14, 2] == 2
      shape: dense_shape == [5 8]
      batch_size: dim of dense_shape == 5
      cols: dim_size(1) of dense_shape == 8
      embedding_size: dim_size(1) of weight tensor == 3
      sparse_gather(input, input_size, indice, indice_dim, shape, batch_size,
                    cols, weight, output, embedding_size, is_mean);
*/

template <typename Device, typename Tid, typename Tshape>
class FusedSafeEmbeddingLookupSparseOp : public OpKernel {
public:
  explicit FusedSafeEmbeddingLookupSparseOp(OpKernelConstruction* context)
           : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("Combiner", &combiner));
    //OP_REQUIRES_OK(context, context->GetAttr("Dims", &dims));
    node_name = context->def().name();

    static bool printed = false;
    if (!printed) {
      printf("******** FusedSafeEmbeddingLookupSparseOp ********\n");
      printed = true;
    }
  }

  ~FusedSafeEmbeddingLookupSparseOp() {
  }

  void Compute(OpKernelContext* context) override {
    // Grab the weight
    float *weight;
    const Tensor* weight_tensor = &context->input(0);

    printf("=============== my fused compute...\n");
    // for saved model
    if (weight_tensor->dtype() == DT_RESOURCE) {
      Var* variable;
      OP_REQUIRES_OK(context,
                     LookupResource(context, HandleFromInput(context, 0), 
                                    &variable));
      core::ScopedUnref s(variable);
      weight_tensor = variable->tensor();
      OP_REQUIRES(context, weight_tensor->dtype() == DT_FLOAT,
                  errors::InvalidArgument("Expect float weight in ",
                                          node_name));
    }

    weight = (float *)weight_tensor->tensor_data().data();
    
    // Input id
    const Tensor& input_tensor = context->input(1);
    Tid *input = (Tid *)input_tensor.tensor_data().data();
    
    const Tensor& shape_tensor = context->input(2);
    Tshape *shape = (Tshape *)shape_tensor.tensor_data().data();

    // To check the input
    OP_REQUIRES(context, (shape_tensor.dims() == 1),
                errors::InvalidArgument("Shape tensor is not valid (dims != 1)"));
    OP_REQUIRES(context, (shape_tensor.dim_size(0) >= 2),
                errors::InvalidArgument("Shape tensor is not valid (dim_size(0) < 2)"));

    int64 input_size = 1;
    for (int i = 0; i < input_tensor.dims(); ++i) {
      input_size *= input_tensor.dim_size(i);
    }
    
    int input_dims = shape_tensor.dim_size(0);
    int cols = shape[input_dims - 1];
    int batch_size = 1;
    for (int i = 0; i < input_dims - 1; ++i) {
      batch_size *= shape[i];
    }
    int embedding_size = weight_tensor->dim_size(1);
    bool is_mean = (combiner == 1);

    const Tensor& indice_tensor = context->input(3);
    Tshape *indice = (Tshape *)indice_tensor.tensor_data().data();
    int indice_dim = indice_tensor.dim_size(1);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape output_shape({batch_size, embedding_size});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    float *output = (float *)output_tensor->tensor_data().data();

    if (input_size == batch_size * cols) { // input id is dense
      sparse_gather(input, batch_size, cols, weight, output, embedding_size, is_mean);
    } else { // input id is sparse
      OP_REQUIRES(context, (indice_tensor.dims() == 2),
                  errors::InvalidArgument("Indice tensor is not as expected (dims != 2)"));
      OP_REQUIRES(context, (indice_tensor.dim_size(0) == input_size),
                  errors::InvalidArgument("Indice tensor is not as expected (dim_size(0) != batch_size)"));
      sparse_gather(input, input_size, indice, indice_dim, shape, batch_size,
                    cols, weight, output, embedding_size, is_mean);
    }

    std::vector<Tid> unique_value; // context->allocate_output(
    unique_value.reserve(input_size);
    std::vector<Tid> unique_indices; // context->allocate_output(
    unique_indices.reserve(input_size);
    for (int64 i = 0; i < input_size; ++i) {
        Tid id = input[i];
        if (id < 0) { // Skip invalid id
          continue;
        }
        auto it = std::find(unique_value.begin(), unique_value.end(), id);
        if (it == unique_value.end()) { // no find
          unique_value.push_back(id);
          unique_indices.push_back(unique_value.size() + 1);
        }
        else {
          unique_indices.push_back(it - unique_value.begin());
        }
    }

    Tensor* unique_value_tensor = NULL;
    TensorShape unique_value_shape({unique_value.size()});
    OP_REQUIRES_OK(context, context->allocate_output(1, unique_value_shape, &unique_value_tensor));
    Tid *output_unique_value = (Tid *)unique_value_tensor->tensor_data().data();
    for (int i = 0; i < unique_value.size(); ++i) {
      output_unique_value[i] = unique_value[i];
    }

    Tensor* unique_indices_tensor = NULL;
    TensorShape unique_indices_shape({unique_indices.size()});
    OP_REQUIRES_OK(context, context->allocate_output(2, unique_indices_shape, &unique_indices_tensor));
    Tid *output_unique_indices = (Tid *)unique_indices_tensor->tensor_data().data();
    for (int i = 0; i < unique_indices.size(); ++i) {
      output_unique_indices[i] = unique_indices[i];
    }

    std::vector<Tid> empty_rows;
    empty_rows.reserve(input_size);
    std::vector<int32> cast_indice; // context->allocate_output(
    cast_indice.reserve(input_size);
    cast_indice.push_back(indice[0]);
    for (int64 i = 1; i < input_size; ++i) {
      if (indice[i * indice_dim] - indice[(i-1) * indice_dim] > 1) {
        int64 start = indice[(i-1) * indice_dim];
        int64 end = indice[i * indice_dim];
        while (end - start > 1) {
          start++;
          cast_indice.push_back(start);
          empty_rows.push_back(start);
        }
      }
      cast_indice.push_back(indice[i * indice_dim]);
    }

    Tensor* cast_indice_tensor = NULL;
    TensorShape cast_indice_shape({cast_indice.size()});
    OP_REQUIRES_OK(context, context->allocate_output(3, cast_indice_shape, &cast_indice_tensor));
    int32 *output_cast_indice = (int32 *)cast_indice_tensor->tensor_data().data();
    for (int i = 0; i < cast_indice.size(); ++i) {
      output_cast_indice[i] = cast_indice[i];
    }

    // TensorShape identity_shape({unique_value.size(), embedding_size}); // context->allocate_output(
    Tensor* identity_shape_tensor = NULL;
    TensorShape identity_shape_shape({2});
    OP_REQUIRES_OK(context, context->allocate_output(4, identity_shape_shape, &identity_shape_tensor));
    int32 *output_identity_shape = (int32 *)identity_shape_tensor->tensor_data().data();
    output_identity_shape[0] = unique_value.size();
    output_identity_shape[1] = embedding_size;
    
    // std::array<std::array<bool, embedding_size>, batch_size> tile;  // context->allocate_output(
    std::unique_ptr<bool[]> tile(new bool[batch_size*embedding_size]);
    for (int64 i = 0; i < batch_size; ++i) {
      if (std::find(empty_rows.begin(), empty_rows.end(), batch_size) == empty_rows.end()) {
        for (int64 j = 0; j < embedding_size; j++) {
          tile[i*embedding_size+j] == false;
        }
      }
      else {
        for (int64 j = 0; j < embedding_size; j++) {
          tile[i*embedding_size+j] == true;
        }
      }
    }

    Tensor* tile_tensor = NULL;
    TensorShape tile_shape({batch_size, embedding_size});
    OP_REQUIRES_OK(context, context->allocate_output(5, tile_shape, &tile_tensor));
    bool *output_tile = (bool *)tile_tensor->tensor_data().data();
    for (int64 i = 0; i < batch_size; ++i) {
      for (int64 j = 0; j < embedding_size; j++) {
        output_tile[i*embedding_size+j] = tile[i*embedding_size+j];
      }
    }

    // TensorShape zeros_like_shape({batch_size, embedding_size}); // context->allocate_output(
    Tensor* zeros_like_tensor = NULL;
    TensorShape zeros_like_shape({batch_size, embedding_size});
    OP_REQUIRES_OK(context, context->allocate_output(6, zeros_like_shape, &zeros_like_tensor));
    float *output_zeros_like = (float *)zeros_like_tensor->tensor_data().data();
    for (int64 i = 0; i < batch_size; ++i) {
      for (int64 j = 0; j < embedding_size; j++) {
        output_tile[i*embedding_size+j] = 0.0;
      }
    }

    Tensor* select_tensor = NULL;
    TensorShape select_shape({2});
    OP_REQUIRES_OK(context, context->allocate_output(7, select_shape, &select_tensor));
    int32 *output_select = (int32 *)select_tensor->tensor_data().data();
    output_select[0] = batch_size;
    output_select[1] = embedding_size;
  }

private:
  // 0=SUM, 1=MEAN
  int combiner;
  std::string node_name;
};

REGISTER_KERNEL_BUILDER(                            \
    Name("FusedSafeEmbeddingLookupSparse")          \
    .Device(DEVICE_CPU)                             \
    .TypeConstraint<int32>("Tid")                   \
    .TypeConstraint<int64>("Tshape"),               \
    FusedSafeEmbeddingLookupSparseOp<CPUDevice, int32, int64>);

REGISTER_KERNEL_BUILDER(                            \
    Name("FusedSafeEmbeddingLookupSparse")          \
    .Device(DEVICE_CPU)                             \
    .TypeConstraint<int64>("Tid")                   \
    .TypeConstraint<int64>("Tshape"),               \
    FusedSafeEmbeddingLookupSparseOp<CPUDevice, int64, int64>);

}  // namespace tensorflow
