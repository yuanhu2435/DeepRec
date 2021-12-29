import tensorflow as tf
from tensorflow.contrib import layers

'''
[array([[ 0.09472656, -0.45898438,  0.56640625],
       [-0.01525879, -0.7265625 , -0.12060547],
       [-0.01525879, -0.7265625 , -0.12060547],
       [ 0.12402344, -0.2578125 ,  0.40039062],
       [ 0.12402344, -0.2578125 ,  0.40039062],
       [-0.01525879, -0.7265625 , -0.12060547],
       [-0.01525879, -0.7265625 , -0.12060547],
       [ 0.12402344, -0.2578125 ,  0.40039062]], dtype=float32)]

[[bfloat16(-0.0152587891) bfloat16(-0.7265625) bfloat16(-0.120605469)]
 [bfloat16(0.124023438) bfloat16(-0.2578125) bfloat16(0.400390625)]
 [bfloat16(-1.03125) bfloat16(0.41015625) bfloat16(-0.703125)]
 [bfloat16(-0.74609375) bfloat16(0.172851562) bfloat16(0.5)]
 [bfloat16(0.0947265625) bfloat16(-0.458984375) bfloat16(0.56640625)]]
'''

'''
def get_tensor_by_name(name: str):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables(),
                                max_to_keep=args.keep_checkpoint_max)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # train model
        sess.run(train_init_op)
        model._is_training = True
        _tensor = sess.graph.get_tensor_by_name(name)
        print("\n", _tensor)
        return sess.run(_tensor)

def get_op_by_name(name: str):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables(),
                                max_to_keep=args.keep_checkpoint_max)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # train model
        sess.run(train_init_op)
        model._is_training = True
        _op = sess.graph.get_operation_by_name(name)
        print("-" * 64, "\n")
        for _input in _op.inputs:
            print(_input)
        print("-" * 64, "\n")

get_op_by_name("dnn/input_from_feature_columns/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/embedding_lookup")

_input_grad = get_tensor_by_name("head/gradients/dnn/input_from_feature_columns/input_layer/C10_embedding/Reshape_grad/Reshape:0")
_output_grad = get_tensor_by_name("head/gradients/dnn/input_from_feature_columns/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/embedding_lookup_grad/Reshape:0")
print("Output:\n", _output_grad.shape, "\n", _output_grad)

_reshape_2_shape_input_tensor = get_tensor_by_name("dnn/input_from_feature_columns/input_layer/C10_embedding/C10_embedding_weights:0")

_zero_like_input_tensor = get_tensor_by_name("dnn/input_from_feature_columns/input_layer/C10_embedding/C10_embedding_weights/zeros_like:0")

_tile_input_tensor = get_tensor_by_name("dnn/input_from_feature_columns/input_layer/C10_embedding/C10_embedding_weights/Tile:0")

_gather_input_tensor = get_tensor_by_name("dnn/input_from_feature_columns/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/embedding_lookup:0")
_unique_1_input_tensor = get_tensor_by_name("dnn/input_from_feature_columns/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/Unique:1")
_cast_input_tensor = get_tensor_by_name("dnn/input_from_feature_columns/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/Cast:0")
_weight_input_tensor = get_tensor_by_name("dnn/input_from_feature_columns/input_layer/C10_embedding/embedding_weights/read:0")
_unique_0_input_tensor = get_tensor_by_name("dnn/input_from_feature_columns/input_layer/C10_embedding/C10_embedding_weights/embedding_lookup_sparse/Unique:0")

_my_grad = _embedding_grad(None, _input_grad, _reshape_2_shape_input_tensor,
                            _zero_like_input_tensor, _tile_input_tensor,
                            _gather_input_tensor, _unique_1_input_tensor,
                            _cast_input_tensor, _weight_input_tensor, 
                            _unique_0_input_tensor)

print("\nmy_grad:\n", _my_grad)
print("\n_output_grad:\n", _output_grad)

check_same_array(_my_grad, _output_grad)


def check_same_array(a, b):
  print("is a == b ? ", (a==b).any())


# @tf.RegisterGradient("Embedding_fusion")
def _embedding_grad(op, grad, _reshape_2_shape_input_tensor,
                        _zero_like_input_tensor, _tile_input_tensor,
                        _gather_input_tensor, _unique_1_input_tensor,
                        _cast_input_tensor, _weight_input_tensor,
                        _unique_0_input_tensor):
    from tensorflow.python.ops import math_ops
    from tensorflow.python.framework import dtypes
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import array_ops
    import warnings

    new_g = tf.Graph()
    with new_g.as_default():
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
        # return [ops.IndexedSlices(values, indices, params_shape), None]
    with tf.Session(graph=new_g) as  sess:
        _rst = sess.run(values)
    return _rst
'''

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import warnings
import tensorflow as tf
import numpy as np

np.random.seed(3)
tf.set_random_seed(1234)

@tf.RegisterGradient("FusedSafeEmbeddingLookupSparse")
def _embedding_grad(op, grad, *all):
    # Reshape_2_grad
    _input0_shape = tf.shape(_reshape_2_shape_input_tensor)
    _reshape_grad = tf.reshape(grad, _input0_shape)
    
    # # Select Grad
    # _zeros_like = tf.zeros_like(_zero_like_input_tensor)
    # _select_grad = tf.where(_tile_input_tensor, _zeros_like, _reshape_grad)

    # # SparseSegmentMeanGrad
    # dim0 = tf.shape(_gather_input_tensor)[0]
    # _sparse_segment_mean_grad = math_ops.sparse_segment_mean_grad(_select_grad, _unique_1_input_tensor, _cast_input_tensor, dim0)

    # # GatherGrad
    # params = _weight_input_tensor
    # with ops.colocate_with(params):
    #     params_shape = array_ops.shape(params, out_type=ops.dtypes.int64)
    #     params_shape = math_ops.cast(params_shape, dtypes.int32)

    # # Build appropriately shaped IndexedSlices
    # indices = _unique_0_input_tensor
    # size = array_ops.expand_dims(array_ops.size(indices), 0)
    # values_shape = array_ops.concat([size, params_shape[1:]], 0)
    # with warnings.catch_warnings():
    #     warnings.filterwarnings(
    #         "ignore",
    #         message="Converting sparse IndexedSlices to a dense Tensor.*")
    #     values = array_ops.reshape(_sparse_segment_mean_grad, values_shape)
    # indices = array_ops.reshape(indices, size)
    # return [ops.IndexedSlices(values, indices, params_shape), None]
    return [grad, None, None, None]


def print_tensor(sess, tensor_name):
  print("\n### name:", tensor_name)
  out = sess.graph.get_tensor_by_name(tensor_name)
  print(out.eval())


def print_ops(sess, op_name, input_dict):
  print("\n### name:", op_name)
  out = sess.graph.get_operation_by_name(op_name)
  print(">" * 64)
  for _input in out.inputs:
    print(_input)
    print(sess.run(_input, feed_dict=input_dict))
  print("-" * 64)
  for _output in out.outputs:
    print(_output)
    print(sess.run(_output, feed_dict=input_dict))
  print("<" * 64, "\n")


def get_model(data):
  inputs = tf.placeholder(dtype=tf.string, name="input")
  features = {}
  columns = []

  with tf.name_scope("embedding"):
    feature_name = 'colors'
    features[feature_name] = inputs
    hash_bucket = tf.feature_column.categorical_column_with_hash_bucket(key=feature_name, hash_bucket_size=5, dtype=tf.string)
    column = tf.feature_column.embedding_column(hash_bucket, 3, combiner='mean')
    # column = tf.feature_column.embedding_column(hash_bucket, 4, do_fusion=True)
    columns.append(column)
    embedding = tf.feature_column.input_layer(features, columns)

  with tf.name_scope("mlp"):
    layer = layers.fully_connected(embedding, 6, activation_fn=tf.nn.leaky_relu)

  labels = tf.constant(1.0, shape=[5, 6], dtype=float)

  with tf.name_scope("logits"):
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer, labels=labels))

  with tf.name_scope("optimizer"):
    train_op = tf.train.AdagradOptimizer(learning_rate=0.01, initial_accumulator_value=0.1).minimize(loss_op)
  return inputs, train_op


def main():
  # colors = {'colors': [['green','red','blue','yellow','pink','blue','red','indigo'], ['','','','','','','',''], ['','','','yellow','pink','blue','red','indigo'], ['','','','','','','',''], ['green','','','','','','','']]}

  colors_arr = ['', 'green', 'red', 'blue', 'yellow', 'pink', 'indigo']
  data = np.random.choice(colors_arr, [5, 8], p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
  print(data)
  data = list(data)

  inputs, train_op = get_model(data)
  init_global = tf.global_variables_initializer()
  init_local = tf.local_variables_initializer()
  init_table = tf.tables_initializer()

  with tf.Session() as sess:
    tf.summary.FileWriter('./graph', sess.graph)
    sess.run([init_global, init_local, init_table])

    input_dict = {}
    input_dict[inputs] = data
    print(sess.run(train_op, feed_dict=input_dict))

    input_tensor_name = ["embedding/input_layer/colors_embedding/lookup",
                        "embedding/input_layer/colors_embedding/to_sparse_input/indices",
                        "input_layer/colors_embedding/embedding_weights/read"]
    temp_tensor_name = ["embedding/input_layer/colors_embedding/colors_embedding_weights/embedding_lookup_sparse/Unique",
                        "embedding/input_layer/colors_embedding/colors_embedding_weights/embedding_lookup_sparse/Cast",
                        "embedding/input_layer/colors_embedding/colors_embedding_weights/embedding_lookup_sparse/embedding_lookup",
                        "embedding/input_layer/colors_embedding/colors_embedding_weights/Tile",
                        "embedding/input_layer/colors_embedding/colors_embedding_weights/zeros_like",
                        "embedding/input_layer/colors_embedding/colors_embedding_weights/Reshape_2",
                        "embedding/input_layer/concat/concat"]
    grad_tensor_name = ["optimizer/gradients/embedding/input_layer/colors_embedding/colors_embedding_weights/embedding_lookup_sparse/embedding_lookup_grad/Reshape",
                     "optimizer/gradients/embedding/input_layer/colors_embedding/colors_embedding_weights/embedding_lookup_sparse_grad/SparseSegmentMeanGrad"]

    for _op in input_tensor_name:
      print_ops(sess, _op, input_dict)

    for _op in temp_tensor_name:
      print_ops(sess, _op, input_dict)

    for _op in grad_tensor_name:
      print_ops(sess, _op, input_dict)

    # saver = tf.train.Saver(max_to_keep=2)
    # saver.save(sess, './ckpt_model/model')

    # converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['mlp/fully_connected/LeakyRelu'])
    # tf.train.write_graph(converted_graph_def, './', "frozen_model.pb", as_text=False)


if __name__ == '__main__':
    main()