# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""ParquetDataset"""

import collections

import tensorflow as tf

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_parquet_ops

class ParquetDataset(dataset_ops.Dataset):
    """ParquetDataset"""

    def __init__(self, filename, columns=None, internal=True):
        """ParquetDataset."""
        assert internal
        with ops.name_scope("ParquetDataset"):
            components, shapes, dt = gen_parquet_ops.parquet_readable_info(
                filename, shared=filename, container="ParquetDataset"
            )

            if not context.executing_eagerly():
                assert columns is not None
                assert isinstance(columns, dict)
                dt = [
                    spec if isinstance(spec, dtypes.DType) else spec.dtype
                    for column, spec in columns.items()
                ]
                columns = list(columns.keys())
            else:
                columns = (
                    None
                    if columns is None
                    else (
                        list(columns.keys()) if isinstance(columns, dict) else columns
                    )
                )

            def shape_f(shapes, components, column):
                shape = array_ops.boolean_mask(shapes, math_ops.equal(components, column))[0]
                shape = array_ops.boolean_mask(shape, math_ops.greater_equal(shape, 0))
                return shape

            def dtype_f(dt, components, column):
                dtype = array_ops.boolean_mask(dt, math_ops.equal(components, column))[0]
                dtype = dtypes.as_dtype(dtype.numpy())
                return dtype

            if columns is not None:
                shapes = [shape_f(shapes, components, column) for column in columns]
                if context.executing_eagerly():
                    dt = [dtype_f(dt, components, column) for column in columns]
                components = columns
            else:
                shapes = ops.unstack(shapes)
                dt = [dtypes.as_dtype(dtype.numpy()) for dtype in ops.unstack(dt)]
                components = [component.numpy() for component in ops.unstack(components)]

            self._filename = filename
            self._components = components
            self._shapes = shapes
            self._dt = dt

            def dataset_f(component, shape, dtype):
                step = 4096
                indices_start = dataset_ops.Dataset.range(0, shape[0], step)
                indices_stop = indices_start.skip(1).concatenate(
                    dataset_ops.Dataset.from_tensor_slices([shape[0]])
                )
                dataset = dataset_ops.Dataset.zip((indices_start, indices_stop))

                def f(start, stop):
                    return gen_parquet_ops.parquet_readable_read(
                        input=self._filename,
                        shared=self._filename,
                        component=component,
                        shape=shape,
                        start=start,
                        stop=stop,
                        dtype=dtype,
                        container="ParquetDataset",
                    )

                dataset = dataset.map(f)
                dataset = dataset.unbatch()
                return dataset

            entries = list(zip(components, shapes, dt))
            datasets = [
                dataset_f(component, shape, dtype)
                for component, shape, dtype in entries
            ]
            self._dataset = dataset_ops.Dataset.zip(
                collections.OrderedDict(list(zip(components, datasets)))
            )

            super().__init__(
               # self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _as_variant_tensor(self):
        return self._dataset._variant_tensor

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec
