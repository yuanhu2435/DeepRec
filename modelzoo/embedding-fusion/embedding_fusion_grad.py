from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import warnings
import tensorflow as tf

@tf.RegisterGradient("Embedding_fusion")
def _embedding_grad(op, grad, _reshape_2_shape_input_tensor,
                        _zero_like_input_tensor, _tile_input_tensor,
                        _gather_input_tensor, _unique_1_input_tensor,
                        _cast_input_tensor, _weight_input_tensor,
                        _unique_0_input_tensor):
    # Reshape_2_grad
    _input0_shape = tf.shape(_reshape_2_shape_input_tensor)
    _reshape_grad = tf.reshape(grad, _input0_shape)
    
    # Select Grad
    _zeros_like = tf.zeros_like(_zero_like_input_tensor)
    _select_grad = tf.where(_tile_input_tensor, _zeros_like, _reshape_grad)

    # SparseSegmentMeanGrad
    dim0 = tf.shape(_gather_input_tensor)[0]
    _sparse_segment_mean_grad = math_ops.sparse_segment_mean_grad(_select_grad, _unique_1_input_tensor, _cast_input_tensor, dim0)

    # GatherGrad
    params = _weight_input_tensor
    with ops.colocate_with(params):
        params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
        params_shape = math_ops.cast(params_shape, dtypes.int32)

    # Build appropriately shaped IndexedSlices
    indices = _unique_0_input_tensor
    size = array_ops.expand_dims(array_ops.size(indices), 0)
    values_shape = array_ops.concat([size, params_shape[1:]], 0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Converting sparse IndexedSlices to a dense Tensor.*")
        values = array_ops.reshape(_sparse_segment_mean_grad, values_shape)
    indices = array_ops.reshape(indices, size)
    return [ops.IndexedSlices(values, indices, params_shape), None]