# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for read_parquet and ParquetDataset."""


import os
import collections
import numpy as np

import tensorflow as tf
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.experimental.ops import parquet_dataset_ops
from tensorflow.python.data.experimental.ops import get_single_element
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

import pandas as pd


# Note: The sample file is generated from the following after apply patch
# tests/test_parquet/parquet_cpp_example.patch:
# `parquet-cpp/examples/low-level-api/reader_writer`
# This test extracts columns of [0, 1, 2, 4, 5]
# with column data types of [bool, int32, int64, float, double].
# Please check `parquet-cpp/examples/low-level-api/reader-writer.cc`
# to find details of how records are generated:
# Column 0 (bool): True for even rows and False otherwise.
# Column 1 (int32): Equal to row_index.
# Column 2 (int64): Equal to row_index * 1000 * 1000 * 1000 * 1000.
# Column 4 (float): Equal to row_index * 1.1.
# Column 5 (double): Equal to row_index * 1.1111111.

class ParquetDatasetTest(test_base.DatasetTestBase):
    @classmethod
    def setUpClass(self):
        filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_parquet",
            "parquet_cpp_example.parquet",
        )
        self.filename = "file://" + filename
        self.filename = "/tmp/parquet_cpp_example.parquet"
        '''
        super(test_base.DatasetTestBase, self).setUp(self)
        self._coord = server_lib.Server.create_local_server()
        self._worker = server_lib.Server.create_local_server()

        self._cluster_def = cluster_pb2.ClusterDef()
        worker_job = self._cluster_def.job.add()
        worker_job.name = 'worker'
        worker_job.tasks[0] = self._worker.target[len('grpc://'):]
        coord_job = self._cluster_def.job.add()
        coord_job.name = 'coordinator'
        coord_job.tasks[0] = self._coord.target[len('grpc://'):]

        session_config = config_pb2.ConfigProto(cluster_def=self._cluster_def)

        self._sess = session.Session(self._worker.target, config=session_config)
        self._worker_device = '/job:' + worker_job.name
        '''

    def test_parquet_graph(self):
        """Test case for parquet in graph mode."""

        # test parquet dataset
        def f(e):
            columns = {
                "boolean_field": dtypes.bool,
                "int32_field": dtypes.int32,
                "int64_field": dtypes.int64,
                "float_field": dtypes.float32,
                "double_field": dtypes.float64,
                "ba_field": dtypes.string,
                "flba_field": dtypes.string,
            }
            dataset = parquet_dataset_ops.ParquetDataset(e, columns)
            dataset = dataset.batch(500)
            return get_single_element.get_single_element(dataset)

        data = f(self.filename)

        for i in range(50):
            v0 = (i % 2) == 0
            v1 = i
            v2 = i * 1000 * 1000 * 1000 * 1000
            v4 = 1.1 * i
            v5 = 1.1111111 * i
            v6 = b"parquet%03d" % i
            v7 = bytearray(b"").join([bytearray((i % 256,)) for _ in range(10)])
            p0 = self.evaluate(data["boolean_field"][i])
            p1 = self.evaluate(data["int32_field"][i])
            p2 = self.evaluate(data["int64_field"][i])
            p4 = self.evaluate(data["float_field"][i])
            p5 = self.evaluate(data["double_field"][i])
            p6 = self.evaluate(data["ba_field"][i])
            p7 = self.evaluate(data["flba_field"][i])

            assert v0 == p0
            assert v1 == p1
            assert v2 == p2
            assert np.isclose(v4, p4)
            assert np.isclose(v5, p5)
            assert v6 == p6
            assert v7 == p7

if __name__ == "__main__":
    test.main()
