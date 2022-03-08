#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace io {
namespace {

REGISTER_OP("ParquetReadableInfo")
    .Input("input: string")
    .Input("shared: string")
    .Attr("container: string = ''")
    .Output("component: string")
    .Output("shape: int64")
    .Output("dtype: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->UnknownDim()}));
      c->set_output(1, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
      c->set_output(2, c->MakeShape({c->UnknownDim()}));
      return Status::OK();
    });

REGISTER_OP("ParquetReadableRead")
    .Input("input: string")
    .Input("shared: string")
    .Input("component: string")
    .Input("shape: int64")
    .Input("start: int64")
    .Input("stop: int64")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Output("value: dtype")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle full;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &full));
      if (!(c->RankKnown(full) && c->Rank(full) > 0)) {
        c->set_output(0, full);
        return Status::OK();
      }
      // TODO: replace dims up until rank(start|stop)
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(c->ReplaceDim(full, 0, c->UnknownDim(), &shape));
      c->set_output(0, shape);
      return Status::OK();
    });

}  // namespace
}  // namespace io
}  // namespace tensorflow
