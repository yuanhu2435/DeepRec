# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for feature_column."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.client import session
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training import adagrad

from tensorflow.python.summary.writer import writer

def _initialized_session(graph=None, config=None):
  sess = session.Session(graph=graph, config=config)
  writer.FileWriter('/home/marvin/DeepRec/graph', sess.graph)
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess

def print_ops(sess, op_name):
  print("\n### name:", op_name)
  out = sess.graph.get_operation_by_name(op_name)
  print(">" * 64)
  for _input in out.inputs:
    print(_input)
    print(sess.run(_input))
  print("-" * 64)
  for _output in out.outputs:
    print(_output)
    print(sess.run(_output))
  print("<" * 64, "\n")


@ops.RegisterGradient("FusedSafeEmbeddingLookupSparse")
def _embedding_grad(op, grad):
    # Reshape_2_grad
    # _input0_shape = tf.shape(_reshape_2_shape_input_tensor)
    # _reshape_grad = tf.reshape(grad, _input0_shape)
    
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
    grad = array_ops.concat([grad, grad], 0)
    return [grad, None, None, None, None]


class EmbeddingColumnTest(test.TestCase):
  
  def _get_model(self, do_fusion=False):
    with ops.name_scope("embedding"):
        features = {
            'colors': [
                        ['green','red','blue','yellow','pink','blue','red','indigo'],
                        ['','','','','','','',''],
                        ['','','','yellow','pink','blue','red','indigo'],
                        ['','','','','','','',''],
                        ['green','','','','','','','']
                    ]
            }
        hash_bucket = fc.categorical_column_with_hash_bucket(key='colors', hash_bucket_size=10)
        embedding_column = fc.embedding_column(hash_bucket, dimension=4, do_fusion=do_fusion)
        _input = fc_old.input_layer(features, [embedding_column])

    labels = constant_op.constant(1.0, shape=[5, 4], dtype=float)

    with ops.name_scope("logits"):
      loss_op = math_ops.reduce_mean(nn_impl.sigmoid_cross_entropy_with_logits(logits=_input, labels=labels))

    with ops.name_scope("optimizer"):
      train_op = adagrad.AdagradOptimizer(learning_rate=0.01, initial_accumulator_value=0.1).minimize(loss_op)

    return train_op

  def _base_model(self, do_fusion=False, check_tensor=[]):
    with ops.Graph().as_default() as graph:
        random_seed.set_random_seed(2021)

        train_op = self._get_model(do_fusion=do_fusion)
        with _initialized_session(graph=graph) as sess:
            # print(">" * 64, "\n")
            # print(sess.run(train_op))
            # print("<" * 64, "\n")
            
            sess.run(train_op)
            
            common_tensor = [
                # "gradients/input_layer/colors_embedding/Reshape_grad/Reshape",
                "input_layer/colors_embedding/embedding_weights/read",
                "embedding/input_layer/concat/concat",
                "embedding/input_layer/colors_embedding/Reshape",
            ]
            for _op_name in common_tensor:
                print_ops(sess, _op_name)
                
            for _op_name in check_tensor:
                print_ops(sess, _op_name)
    pass

  @test_util.run_deprecated_v1
  def test_fusion_embedding_origin(self):
    check_tensor = [
            # "input_layer/colors_embedding/colors_embedding_weights/embedding_lookup_sparse/embedding_lookup",
        ]
    # self._base_model(check_tensor=check_tensor)

  @test_util.run_deprecated_v1
  def test_fusion_embedding_fusion(self):
    check_tensor = [
            # "gradients/input_layer/colors_embedding/Reshape_grad/Reshape",
            "embedding/input_layer/colors_embedding/colors_embedding_weights/FusedSafeEmbeddingLookupSparse",
        ]
    self._base_model(True, check_tensor=check_tensor)


if __name__ == '__main__':
  test.main()
