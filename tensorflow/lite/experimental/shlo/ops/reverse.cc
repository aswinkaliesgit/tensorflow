
#include "tensorflow/lite/experimental/shlo/ops/reverse.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

absl::Status CheckParameters(
    // constraint 1
    const Tensor& operand,
    absl::InlinedVector<Axis, kMaxNumDimensions>& dimensions, Tensor& result) {
  if (operand.IsQuantized()) {
    SHLO_REF_RETURN_ON_ERROR(
        CheckSameBaselineType(CheckCtx("reverse"), operand, result));

  } else {
    if (operand.tensor_element_type() != result.tensor_element_type()) {
      return absl::FailedPreconditionError(
          "The element type of operand and result must be same.");
    }
  }
  // constraint 2
  bool map[] = {0, 0, 0, 0, 0, 0};
  for (auto d : dimensions) {
    if (d >= 6) {
      return absl::FailedPreconditionError("Dimension out of range.");
    }
    if (map[d] != 0) {
      return absl::FailedPreconditionError(
          "Every dimension to be reversed must be unique.");
    } else
      map[d] = 1;
  }

  // constraint 3
  for (auto d : dimensions) {
    if (d < 0 || d >= result.Rank()) {
      return absl::FailedPreconditionError(
          "The dimensions must be in the range of the rank of result.");
    }
  }
  return absl::OkStatus();
}

template <DataType storage_type>
absl::Status PrepareTensorsQuantized(reverseOp& op, const Tensor& operand,
                                     Tensor& output) {
  using StorageT = StorageType<storage_type>;
  const DimensionSize operand_size = operand.NumElements();
  const DimensionSize output_size = output.NumElements();
  op.operand_dequantized_data =
      std::vector<std::byte>(operand_size * sizeof(StorageT));
  const Shape operand_shape = operand.shape();
  Tensor operand_dequantized{
      .type = TensorType{.shape = operand_shape, .element_type = storage_type},
      .data = op.operand_dequantized_data.data()};
  op.output_dequantized_data =
      std::vector<std::byte>(output_size * sizeof(StorageT));
  const Shape output_dequantized_shape = output.shape();
  Tensor output_dequantized{
      .type = TensorType{.shape = output_dequantized_shape,
                         .element_type = storage_type},
      .data = op.output_dequantized_data.data()};

  op.operand_dequantized = std::move(operand_dequantized);
  op.output_dequantized = std::move(output_dequantized);

  return absl::OkStatus();
}
template <DataType storage_type>
absl::Status EvaluateImpl(
    const Tensor& operand,
    absl::InlinedVector<Axis, kMaxNumDimensions>& dimensions, Tensor& output) {
  using StorageT = StorageType<storage_type>;

  const StorageT* operand_buffer = operand.GetDataAs<storage_type>();
  StorageT* output_buffer = output.GetDataAs<storage_type>();
  const DimensionSize operand_size = operand.NumElements();
  const DimensionSize output_size = output.NumElements();
  const size_t operand_rank = operand.Rank();
  const size_t output_rank = output.Rank();

  DimensionSize operand_element_index = 0, output_element_index = 0;
  DimensionSize operand_dim_accumulator = 1, output_dim_accumulator = 1;
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> operand_index_helper;
  operand_index_helper.resize(operand_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_index_helper;
  output_index_helper.resize(output_rank);
  absl::InlinedVector<Axis, kMaxNumDimensions> operand_index;
  operand_index.resize(operand_rank);
  absl::InlinedVector<Axis, kMaxNumDimensions> result_index;
  result_index.resize(output_rank);

  for (size_t i = 0; i < operand_rank; ++i) {
    operand_dim_accumulator *= operand.shape().Dim(i);
    operand_index_helper[i] = operand_size / operand_dim_accumulator;
    output_dim_accumulator *= output.shape().Dim(i);
    output_index_helper[i] = output_size / output_dim_accumulator;
  }

  for (size_t k = 0; k < operand_size; ++k) {
    operand.GetNdIndex(k, operand_index);
    absl::c_fill(result_index, 0);
    for (size_t d = 0; d < output_rank; ++d) {
      if (std::find(dimensions.begin(), dimensions.end(), d) !=
          dimensions.end()) {
        result_index[d] = output.shape().Dim(d) - operand_index[d] - 1;
      } else {
        result_index[d] = operand_index[d];
      }
    }
    output_buffer[output.FlattenIndex(result_index)] =
        operand.Get<storage_type>(operand_index);
  }
  return absl::OkStatus();
}

template <DataType storage_type, DataType expressed_type>
void DequantizeOpQuantizePerTensor(reverseOp& op, const Tensor& operand,
                                   Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;

  const StorageT* operand_data = operand.GetDataAs<storage_type>();
  ExpressedT* operand_dequantized_data =
      op.operand_dequantized.GetDataAs<expressed_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  ExpressedT* output_dequantized_data =
      op.output_dequantized.GetDataAs<expressed_type>();

  const DimensionSize operand_num_elements = operand.NumElements();
  const StorageT operand_zero_point =
      operand.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT operand_scale =
      operand.quantized_per_tensor_element_type().ScaleAs<expressed_type>();

  for (DimensionSize i = 0; i < operand_num_elements;
       ++i, ++operand_data, ++operand_dequantized_data) {
    *operand_dequantized_data =
        Dequantize(*operand_data, operand_zero_point, operand_scale);
  }

  absl::Status status =
      Evaluate(op, op.operand_dequantized, op.output_dequantized);

  const DimensionSize output_num_elements = output.NumElements();
  const StorageT output_zero_point =
      output.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT output_scale =
      output.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const ExpressedT inv_scale = static_cast<ExpressedT>(1 / output_scale);

  for (DimensionSize i = 0; i < output_num_elements;
       ++i, ++output_dequantized_data, ++output_data) {
    *output_data = Quantize<storage_type, expressed_type>(
        *output_dequantized_data, output_zero_point, inv_scale);
  }
}

reverseOp Create(reverseOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(reverseOp& op, const Tensor& operand, Tensor& result) {
  if (absl::Status status =
          CheckParameters(operand, op.attributes.dimensions, result);
      !status.ok()) {
    return status;
  }

  SHLO_REF_RETURN_ON_ERROR(
      CheckParameters(operand, op.attributes.dimensions, result));
  if (operand.IsQuantized()) {
    DISPATCH_BOOL_INT_FLOAT(
        PrepareTensorsQuantized,
        operand.quantized_per_tensor_element_type().ExpressedType(), op,
        operand, result);
  }
  return absl::OkStatus();
}

absl::Status Evaluate(reverseOp& op, const Tensor& operand, Tensor& result) {
  if (operand.IsQuantized()) {
    if (operand.IsPerTensorQuantized()) {
      DISPATCH_QUANTIZED(
          DequantizeOpQuantizePerTensor,
          operand.quantized_per_tensor_element_type().StorageType(),
          operand.quantized_per_tensor_element_type().ExpressedType(), op,
          operand, result);
    }
  }

  DISPATCH_BOOL_INT_FLOAT(EvaluateImpl, result.tensor_element_type(), operand,
                          op.attributes.dimensions, result);
  return absl::FailedPreconditionError(
      "stablehlo.dot_general: Unsupported tensor type.");
}
}  // namespace shlo_ref
