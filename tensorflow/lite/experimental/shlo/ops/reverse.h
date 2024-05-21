#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_REVERSE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_REVERSE_H_

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

class reverseOp {
 public:
  struct Attributes {
    absl::InlinedVector<Axis, kMaxNumDimensions>& dimensions;
  };
  Attributes attributes;
  Tensor operand_dequantized;
  Tensor output_dequantized;
  std::vector<std::byte> operand_dequantized_data;
  std::vector<std::byte> output_dequantized_data;
};

reverseOp Create(reverseOp::Attributes);

absl::Status Prepare(reverseOp& op, const Tensor& operand, Tensor& result);

absl::Status Evaluate(reverseOp& op, const Tensor& operand, Tensor& result);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_TRANSPOSE_H_