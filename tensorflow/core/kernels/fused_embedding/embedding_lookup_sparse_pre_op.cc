#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/bounds_check.h"

namespace tensorflow {

struct IndicePair {
  int64_t row;
  int64_t column;
};

typedef Eigen::ThreadPoolDevice CPUDevice;

class FusedEmbeddingSparsePreLookUpCPU : public OpKernel {
 public:
  explicit FusedEmbeddingSparsePreLookUpCPU(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_strategy", &partition_strategy_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partition_axis", &partition_axis_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_empty_row", &fill_empty_row_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("prune_invalid_id", &prune_invalid_id_));
    int temp_default_id;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_id", &temp_default_id));
    default_id_ = int64_t(temp_default_id);
  }

  void Compute(OpKernelContext* ctx) override {

    const int64_t default_id = default_id_ >= 0 ? default_id_ : 0;
    // 1. bind inputs
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

    partition_sizes_accumulate_.clear();
    for (const Tensor& shape : partition_shapes) {
      OP_REQUIRES(ctx, shape.dims() <= 2,
                  errors::InvalidArgument(
                      "input partition_shapes must all less than rank 2"));
      const int64_t accu = partition_sizes_accumulate_.empty()
                               ? shape.flat<int64>().data()[0]
                               : shape.flat<int64>().data()[0] +
                                     partition_sizes_accumulate_.back();
      partition_sizes_accumulate_.push_back(accu);
    }

    // 2.1 get index
    std::set<int64_t> indices_set;
    std::vector<std::vector<IndicePair>> new_index_(num_partitions_, std::vector<IndicePair>(0));

    std::vector<IndicePair> fill_empty_row_index_;
    int64_t fill_empty_row_p_seg_ = 0;
    int64_t fill_empty_row_p_val_ = 0;

    int64_t p_seg = 0;
    int64_t p_val = 0;
    int64_t tmp_value = 0;
    const int64_t total_size = partition_sizes_accumulate_.back();
    const int64_t ids_per_partition = total_size / num_partitions_;
    const int64_t extras = total_size % num_partitions_;

    for (int64_t origin_index = 0; origin_index < nnz; ++origin_index) {
      tmp_value = values[origin_index];
      if (tmp_value < 0){
        if(prune_invalid_id_) continue;
        p_seg = 0;
        p_val = tmp_value;
      } else {
        if(partition_strategy_ == "mod"){
          p_seg = tmp_value % num_partitions_;
          p_val = tmp_value / num_partitions_;
        }
        if(partition_strategy_ == "div"){
          p_seg = tmp_value < extras * (ids_per_partition + 1) ?
                    tmp_value / (ids_per_partition + 1) :
                    (tmp_value - extras) / ids_per_partition;
          p_val = p_seg < extras ?
                    tmp_value % (ids_per_partition + 1) :
                    (tmp_value - extras) % ids_per_partition;
        }
      }

      new_index_[p_seg].push_back({origin_index, p_val});
      indices_set.insert(indices[origin_index].row);
    }
    
    for (int64_t origin_index = 0; fill_empty_row_ && origin_index < batch_size; ++origin_index){
      if(indices_set.count(origin_index)) continue;

      tmp_value = default_id;
      if(partition_strategy_ == "mod"){
        fill_empty_row_p_seg_ = tmp_value % num_partitions_;
        fill_empty_row_p_val_ = tmp_value / num_partitions_;
      }
      if(partition_strategy_ == "div"){
        fill_empty_row_p_seg_ = tmp_value < extras * (ids_per_partition + 1) ?
                  tmp_value / (ids_per_partition + 1) :
                  (tmp_value - extras) / ids_per_partition;
        fill_empty_row_p_val_ = p_seg < extras ?
                  tmp_value % (ids_per_partition + 1) :
                  (tmp_value - extras) % ids_per_partition;
      }

      fill_empty_row_index_.push_back({origin_index, 0});
    }

    // std::cout << "fill_empty_row_p_val_ = " << fill_empty_row_p_val_ << std::endl;
    // std::cout << "fill_empty_row_p_seg_ = " << fill_empty_row_p_seg_ << std::endl;

    OpOutputList partitioned_values;
    OP_REQUIRES_OK(ctx,
                   ctx->output_list("partitioned_values", &partitioned_values));
    OpOutputList partitioned_indices;
    OP_REQUIRES_OK(
        ctx, ctx->output_list("partitioned_indices", &partitioned_indices));
    

    for (int i = 0; i < num_partitions_; ++i) {
      int64_t size = new_index_[i].size();
      if (fill_empty_row_ && i == fill_empty_row_p_seg_){
        size += fill_empty_row_index_.size();
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

      for (int j = 0; j < new_index_[i].size(); ++j){
        sub_partitioned_values_data[j] = new_index_[i][j].column;
        sub_partitioned_indices_data[j] = indices[new_index_[i][j].row];
      }

      if (fill_empty_row_ && i == fill_empty_row_p_seg_){
        for (int j = 0, l = new_index_[i].size(); j < fill_empty_row_index_.size(); ++j, ++l){
          sub_partitioned_values_data[l] = fill_empty_row_p_val_;
          sub_partitioned_indices_data[l] = fill_empty_row_index_[j];
        }
      }
    }
  }

 private:
  int num_partitions_;
  int partition_axis_;
  bool fill_empty_row_;
  bool prune_invalid_id_;
  int64_t default_id_;
  std::string partition_strategy_;
  std::vector<int64_t> partition_sizes_accumulate_;
  std::vector<int64_t> invalid_ids_list_;
  std::vector<int64_t> empty_index_list_;
};


REGISTER_KERNEL_BUILDER(                                         \
    Name("FusedSafeEmbeddingPreLookup")                          \
    .Device(DEVICE_CPU)                                          \
    .HostMemory("partition_shapes")                              \
    .HostMemory("sp_dense_shape"),                               \
    FusedEmbeddingSparsePreLookUpCPU);
}  // namespace tensorflow
