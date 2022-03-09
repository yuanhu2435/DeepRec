#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/bounds_check.h"

namespace tensorflow {

namespace{

struct IndicePair {
  int64_t row;
  int64_t column;
};

enum Part_Strategy {
  MOD,
  DIV
};

typedef void (*PARTITIONALGO)(const int64_t, const int64_t,
                              const int64_t, const int64_t,
                              const int64_t, const int64_t,
                              const int64_t, int64_t&, int64_t&);

template <Part_Strategy PS>
inline void GetPartitionIndex(const int64_t numPartitions, const int64_t totalSize,
                       const int64_t idsPerPartition, const int64_t extras,
                       const int64_t idsPerPartitionPlus, const int64_t idSubxtras,
                       const int64_t originId, int64_t& segment, int64_t& newId){
  // OP_REQUIRES(ctx, false,
  //   errors::InvalidArgument("GetPartitionIndex not support undefine type. ", T));
  //todo(marvin): show the error info.
}

template <>
inline void GetPartitionIndex<Part_Strategy::MOD>(
                        const int64_t numPartitions, const int64_t totalSize,
                        const int64_t idsPerPartition, const int64_t extras,
                        const int64_t idsPerPartitionPlus, const int64_t idSubxtras,
                        const int64_t originId, int64_t& segment, int64_t& newId){
  segment = originId % numPartitions;
  newId = originId / numPartitions;
}

template <>
inline void GetPartitionIndex<Part_Strategy::DIV>(
                        const int64_t numPartitions, const int64_t totalSize,
                        const int64_t idsPerPartition, const int64_t extras,
                        const int64_t idsPerPartitionPlus, const int64_t idSubxtras,
                        const int64_t originId, int64_t& segment, int64_t& newId){
  // segment = originId < extras * (idsPerPartition + 1) ?
  //           originId / (idsPerPartition + 1) :
  //           (originId - extras) / idsPerPartition;
  // newId = segment < extras ?
  //           originId % (idsPerPartition + 1) :
  //           (originId - extras) % idsPerPartition;

  register int64_t p_seg_0, p_seg_1, p_val_0, p_val_1;
  register bool x, y;

  p_seg_0 = originId / idsPerPartitionPlus;
  p_seg_1 = idSubxtras / idsPerPartition;

  p_val_0 = originId % idsPerPartitionPlus;
  p_val_1 = idSubxtras % idsPerPartition;

  x = extras && !(originId / (extras * idsPerPartitionPlus));
  segment = x * p_seg_0 + !x * p_seg_1;
  y = extras && !((x * p_seg_0 + !x * p_seg_1) / extras);
  newId = y * p_val_0 + !y * p_val_1;
}

template <typename T>
void ShowLog(const std::chrono::time_point<T>& start, const std::string& msg = "") {
  auto end = std::chrono::high_resolution_clock::now();
  VLOG(1) << ">>>" << " time= "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us; message=" << msg;
}
}

typedef Eigen::ThreadPoolDevice CPUDevice;

class FusedEmbeddingSparsePreLookUpCPU : public OpKernel {
 public:
  explicit FusedEmbeddingSparsePreLookUpCPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row", &fill_empty_row_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("prune_invalid_id", &prune_invalid_id_));

    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_strategy", &partition_strategy_str_));
    if(partition_strategy_str_ == "div"){
      partition_strategy_ = GetPartitionIndex<Part_Strategy::DIV>;
    } else if(partition_strategy_str_ == "mod"){
      partition_strategy_ = GetPartitionIndex<Part_Strategy::MOD>;
    } else {
      OP_REQUIRES(ctx, false,
        errors::InvalidArgument("Not support partition_strategy type. ", partition_strategy_));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    auto start = std::chrono::high_resolution_clock::now();
    ShowLog(start, "start Computing");

    const int64_t default_id = default_id_ >= 0 ? default_id_ : 0;
    // 1. get input tensor
    Tensor const* values_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_values", &values_tensor));
    const int64_t nnz = values_tensor->shape().dim_size(0);

    const int64_t* values = reinterpret_cast<const int64_t*>(
                                  values_tensor->flat<int64>().data());

    Tensor const* indices_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_indices", &indices_tensor));

    const IndicePair* indices = reinterpret_cast<const IndicePair*>(
                                  indices_tensor->flat<int64>().data());

    Tensor const* dense_shape = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sp_dense_shape", &dense_shape));
    const int64_t batch_size = dense_shape->flat<int64>().data()[0];

    OpInputList partition_shapes;
    OP_REQUIRES_OK(ctx, ctx->input_list("partition_shapes", &partition_shapes));

    partition_total_sizes_ = 0;
    for (const Tensor& shape : partition_shapes) {
      OP_REQUIRES(ctx, shape.dims() <= 2,
                  errors::InvalidArgument(
                      "input partition_shapes must all less than rank 2"));
      partition_total_sizes_ += shape.flat<int64>().data()[0];
    }

    // fixme(marvin): show error info when got fake input.
    OP_REQUIRES(ctx, partition_total_sizes_ != 1,
        errors::InvalidArgument("Not support EV yet"));

    // 1.1 define output tensors
    OpOutputList partitioned_values;
    OP_REQUIRES_OK(ctx,
                   ctx->output_list("partitioned_values", &partitioned_values));
    OpOutputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->output_list("partitioned_indices", &partitioned_indices));

    Tensor* all_flags;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(2 * num_partitions_,
                                  TensorShape{batch_size + nnz}, &all_flags));
    int32_t* all_flags_list = all_flags->flat<int32_t>().data();

    memset(all_flags_list, 0, (batch_size + nnz) * sizeof(int32_t));
    ShowLog(start, "// 1.1 define output tensors");

    // 2.1 get index
    const int64_t idsPerPartition = partition_total_sizes_ / num_partitions_;
    const int64_t idsPerPartitionPlus = idsPerPartition + 1;
    const int64_t extras = partition_total_sizes_ % num_partitions_;
    std::vector<IndicePair> empty_index_;
    int64_t* id_index_array = new int64_t[num_partitions_ + nnz * 2];
    // memset(id_index_array, 0, (num_partitions_ + nnz * 2) * sizeof(int64_t));
    memset(id_index_array, 0, (num_partitions_) * sizeof(int64_t));
    ShowLog(start, "// 1.2 memset output");
    int64_t fill_empty_row_p_seg_ = 0;
    int64_t fill_empty_row_p_val_ = 0;
    int64_t p_seg = 0;
    int64_t p_val = 0;

    // 2.2 get the map of the mutli-table index
#ifdef __AVX512F__

#else
    register int64_t tmp_value;
    for (int64_t index = 0, id_index = num_partitions_; index < nnz; ++index, ++id_index) {
      tmp_value = values[index];
      if (tmp_value < 0){
        if (prune_invalid_id_){
          p_seg = -1;
          p_val = tmp_value;
        } else {
          p_seg = 0;
          p_val = tmp_value;
          ++id_index_array[p_seg];
          ++all_flags_list[indices[index].row];
        }
      } else {
        all_flags_list[batch_size + index] = 1;
        ++all_flags_list[indices[index].row];
        //fixme(marvin): How to use macro the instead of the func call?
        partition_strategy_(num_partitions_, partition_total_sizes_,
                            idsPerPartition, extras,
                            idsPerPartitionPlus, tmp_value - extras,
                            tmp_value, p_seg, p_val);
        ++id_index_array[p_seg];
      }
      id_index_array[id_index] = p_seg;
      id_index_array[id_index + nnz] = p_val;
    }
#endif
    ShowLog(start, "// 2.2 get the map of the mutli-table index");

    // 2.3 fill_empty_row_index_
    if (fill_empty_row_){
      // get default id p_seg_ and p_val_
      partition_strategy_(num_partitions_, partition_total_sizes_,
                          idsPerPartition, extras, idsPerPartitionPlus,
                          default_id - extras, default_id,
                          fill_empty_row_p_seg_, fill_empty_row_p_val_);

      for (int64_t origin_index = 0; origin_index < batch_size; ++origin_index){
        if(all_flags_list[origin_index]){
          all_flags_list[origin_index] = 0;
          continue;
        }
        all_flags_list[origin_index] = 1;
        empty_index_.push_back({origin_index, 0});
      }
    }
    ShowLog(start, "// 2.3 fill_empty_row_index_");

    // 3 packaging the output tensor
    for (int i = 0; i < num_partitions_; ++i) {
      int64_t size = id_index_array[i];
      if(fill_empty_row_ && i == fill_empty_row_p_seg_){
        size += empty_index_.size();
      }

      Tensor* sub_partitioned_values;
      OP_REQUIRES_OK(ctx, partitioned_values.allocate(
                              i, TensorShape({static_cast<int64_t>(size)}),
                              &sub_partitioned_values));
      int64_t* sub_partitioned_values_data = reinterpret_cast<int64_t*>(
          sub_partitioned_values->flat<int64>().data());

      Tensor* sub_partitioned_indices;
      OP_REQUIRES_OK(ctx, partitioned_indices.allocate(
                              i, TensorShape({static_cast<int64_t>(size), 2}),
                              &sub_partitioned_indices));

      IndicePair* sub_partitioned_indices_data = reinterpret_cast<IndicePair*>(
                                  sub_partitioned_indices->flat<int64>().data());
      if (!size) continue;
      
      int sub_partitioned_index = 0;
      for (int index = 0; index < nnz; ++index){
        if (id_index_array[index + num_partitions_] == i){
          sub_partitioned_values_data[sub_partitioned_index] = id_index_array[index + num_partitions_ + nnz];
          sub_partitioned_indices_data[sub_partitioned_index] = indices[index];
          ++sub_partitioned_index;
        }
      }
      if(fill_empty_row_ && fill_empty_row_p_seg_ == i){
        memcpy(sub_partitioned_indices_data + sub_partitioned_index,
          empty_index_.data(), empty_index_.size() * sizeof(IndicePair));

        std::fill(sub_partitioned_values_data + sub_partitioned_index,
          sub_partitioned_values_data + size, fill_empty_row_p_val_);
      }
    }
    ShowLog(start, "// 3 packaging the output tensor");
    delete[] id_index_array;
    ShowLog(start, "// 4 delete array");
  }

 private:
  int num_partitions_;
  int partition_total_sizes_;
  int partition_axis_;
  bool fill_empty_row_;
  bool prune_invalid_id_;
  int64_t default_id_;
  PARTITIONALGO partition_strategy_;
  std::string partition_strategy_str_;
};

REGISTER_KERNEL_BUILDER(                                         \
    Name("FusedEmbeddingSparsePreLookUp")                        \
    .Device(DEVICE_CPU)                                          \
    .HostMemory("partition_shapes")                              \
    .HostMemory("sp_dense_shape"),                               \
    FusedEmbeddingSparsePreLookUpCPU);
}  // namespace tensorflow
