#include "tensorflow/lite/experimental/shlo/ops/reverse.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/i4.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using shlo_ref::testing::StatusIs;
using testing::ElementsAreArray;
using testing::Eq;
using testing::FloatEq;
using testing::NanSensitiveFloatEq;
using testing::Pointwise;
namespace shlo_ref {

namespace {
template <class T>
struct NonQuantizedIntReverseTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedIntReverseTest, IntTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedIntReverseTest, IntTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({3, 2});
  const Shape shapeR({3, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{(1), (2), (3), (4), (5), (6)};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0, 1};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data =
      Vector<StorageT>{(6), (5), (4), (3), (2), (1)};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, expected_data);
}

using kSI32TestTypes = ::testing::Types<TestParam<DataType::kSI32>>;
template <class T>
struct NonQuantizedkSI32ReverseTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkSI32ReverseTest, kSI32TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkSI32ReverseTest, kSI32TestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({4, 5});
  const Shape shapeR({4, 5});
  Vector<StorageT> operand_data = Vector<StorageT>{
      -1, -5, 2, 1, 0, 0, -2, 1, -1, 0, 5, -3, -1, 1, 1, 2, 0, 0, 3, -1};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data = Vector<StorageT>{
      2, 0, 0, 3, -1, 5, -3, -1, 1, 1, 0, -2, 1, -1, 0, -1, -5, 2, 1, 0};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, expected_data);

}  // rev_dtypes_shape_int32_4_5__dimensions__0.mlir

using kSI16TestTypes = ::testing::Types<TestParam<DataType::kSI16>>;
template <class T>
struct NonQuantizedkSI16ReverseTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkSI16ReverseTest, kSI16TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkSI16ReverseTest, kSI16TestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({4, 5});
  const Shape shapeR({4, 5});
  Vector<StorageT> operand_data = Vector<StorageT>{
      1, 0, 1, 3, 0, 0, 3, 0, 0, -2, 4, -3, 5, 0, 5, 4, -3, 6, -3, 0};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data = Vector<StorageT>{
      4, -3, 6, -3, 0, 4, -3, 5, 0, 5, 0, 3, 0, 0, -2, 1, 0, 1, 3, 0};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, expected_data);

}  // rev_dtypes_shape_int16_4_5__dimensions__0.mlir

using kSI8TestTypes = ::testing::Types<TestParam<DataType::kSI8>>;
template <class T>
struct NonQuantizedkSI8ReverseTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkSI8ReverseTest, kSI8TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkSI8ReverseTest, kSI8TestTypesTensorsWork) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({4, 5});
  const Shape shapeR({4, 5});
  Vector<StorageT> operand_data = Vector<StorageT>{
      -7, 1, 2, 2, 1, -4, -3, -1, 2, 1, 0, 1, 0, -2, 2, -1, 0, -1, 1, 0};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data = Vector<StorageT>{
      -1, 0, -1, 1, 0, 0, 1, 0, -2, 2, -4, -3, -1, 2, 1, -7, 1, 2, 2, 1};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, expected_data);
}  // rev_dtypes_shape_int8_4_5__dimensions__0.mlir

template <class T>
struct QuantizedIntReverseTest : ::testing::Test {};
TYPED_TEST_SUITE(QuantizedIntReverseTest, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedIntReverseTest, QuantizedTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;
  const Shape shapeOperand({3, 2});
  const Shape shapeR({3, 2});

  Vector<int32_t> operand_data_int =
      Vector<int32_t>{(1), (2), (3), (4), (5), (6)};
  Vector<StorageT> operand_data(operand_data_int.begin(),
                                operand_data_int.end());
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0, 1};
  Vector<StorageT> output_data(shapeR.NumElements());
  const ExpressedT scale = static_cast<ExpressedT>(2);
  const StorageT zero_point = static_cast<StorageT>(0);

  const QuantizedElementTypePerTensor tensor_type =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);

  Tensor operand{
      .type = QuantizedPerTensorTensorType{.shape = shapeOperand,
                                           .element_type = tensor_type},
      .data = operand_data.data()};
  Tensor output{
      .type = QuantizedPerTensorTensorType{.shape = shapeR,
                                           .element_type = tensor_type},
      .data = output_data.data()};
  auto op = Create(reverseOp::Attributes{.dimensions = dimensions});

  Vector<float> expected_data = Vector<float>{(6), (5), (4), (3), (2), (1)};
  Vector<float> expected_quantized(shapeR.NumElements());
  std::transform(expected_data.begin(), expected_data.end(),
                 expected_quantized.begin(), [&](float val) {
                   return Quantize<TypeParam::kStorage, TypeParam::kExpressed>(
                       static_cast<ExpressedT>(val), zero_point,
                       static_cast<ExpressedT>(1.0) / scale);
                 });
  ASSERT_OK(Prepare(op, operand, output));
  ASSERT_OK(Evaluate(op, operand, output));
  EXPECT_THAT(output_data, Pointwise(Eq(), expected_data));
}

using kF32TestTypes = ::testing::Types<TestParam<DataType::kF32>>;
template <class T>
struct NonQuantizedkF32ReverseTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkF32ReverseTest, kF32TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkF32ReverseTest, kF32TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({3, 4, 5});
  const Shape shapeR({3, 4, 5});
  Vector<StorageT> operand_data{
      -1.39323878,  -2.07355523,  3.61496592,  -1.41638482,  4.3499279,
      4.61657476,   -0.847035408, 0.39680019,  4.4041729,    3.43903923,
      -1.1432842,   2.33014345,   -4.82425261, -1.52138329,  8.390310e+00,
      0.895365178,  -1.7085067,   -1.49679315, 0.981733798,  -2.47507167,
      -3.56485152,  -2.95351219,  1.17888641,  1.69931138,   -0.0145214852,
      1.68052375,   2.70574522,   -1.23294461, -1.20053291,  -3.10411549,
      -0.671810328, 2.47057939,   1.90365231,  -0.240815163, -5.70334673,
      5.26833439,   -2.79723477,  -2.1762886,  -1.09088278,  -0.494020909,
      2.68829536,   1.48864734,   -2.68438172, 4.32412481,   -6.42869281,
      5.74774504,   0.600558162,  -3.89856243, -2.57673311,  2.84599566,
      2.61949801,   -1.06490338,  3.48048162,  -2.98022199,  0.0734020919,
      2.57875085,   3.73390079,   -3.21010566, 2.22122025,   -3.73207211};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0, 1, 2};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data{
      -3.73207211,   2.22122025,   -3.21010566, 3.73390079,   2.57875085,
      0.0734020919,  -2.98022199,  3.48048162,  -1.06490338,  2.61949801,
      2.84599566,    -2.57673311,  -3.89856243, 0.600558162,  5.74774504,
      -6.42869281,   4.32412481,   -2.68438172, 1.48864734,   2.68829536,
      -0.494020909,  -1.09088278,  -2.1762886,  -2.79723477,  5.26833439,
      -5.70334673,   -0.240815163, 1.90365231,  2.47057939,   -0.671810328,
      -3.10411549,   -1.20053291,  -1.23294461, 2.70574522,   1.68052375,
      -0.0145214852, 1.69931138,   1.17888641,  -2.95351219,  -3.56485152,
      -2.47507167,   0.981733798,  -1.49679315, -1.7085067,   0.895365178,
      8.390310e+00,  -1.52138329,  -4.82425261, 2.33014345,   -1.1432842,
      3.43903923,    4.4041729,    0.39680019,  -0.847035408, 4.61657476,
      4.3499279,     -1.41638482,  3.61496592,  -2.07355523,  -1.39323878};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkF32ReverseTest, kF32TestTypesTensorsWork2) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({3, 4, 5});
  const Shape shapeR({3, 4, 5});
  Vector<StorageT> operand_data{
      -1.98385429,  -2.30945373,  1.51759779,   -0.285959214, 2.47747445,
      0.272397488,  -3.93382144,  -1.74110687,  -2.28210092,  -1.16285217,
      -0.794024169, 6.73959827,   5.71728086,   9.38643741,   -1.17788923,
      3.13179207,   1.20094347,   2.41477132,   -2.78201818,  2.3772893,
      3.06073213,   -2.7227149,   0.123076648,  -4.37336493,  -6.92281866,
      -5.54955769,  -4.16841936,  -3.15850592,  -3.69914746,  0.840284943,
      2.41977453,   -3.59469795,  -3.00077724,  0.41007033,   7.05760241,
      1.72370541,   -2.81880975,  -1.63942921,  -0.33505857,  0.445375204,
      -2.49106288,  6.38465643,   -2.12248063,  2.30540729,   3.12124443,
      -1.88724816,  3.67404461,   2.30696249,   -0.936812341, -2.2690382,
      -4.78598976,  -0.919804751, -0.780682444, 5.88345575,   -2.79272985,
      3.88456535,   3.04881573,   0.686845183,  -2.81020141,  4.195130e+00};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0, 2};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data{
      3.12124443,   2.30540729,   -2.12248063,  6.38465643,   -2.49106288,
      -2.2690382,   -0.936812341, 2.30696249,   3.67404461,   -1.88724816,
      -2.79272985,  5.88345575,   -0.780682444, -0.919804751, -4.78598976,
      4.195130e+00, -2.81020141,  0.686845183,  3.04881573,   3.88456535,
      -6.92281866,  -4.37336493,  0.123076648,  -2.7227149,   3.06073213,
      0.840284943,  -3.69914746,  -3.15850592,  -4.16841936,  -5.54955769,
      7.05760241,   0.41007033,   -3.00077724,  -3.59469795,  2.41977453,
      0.445375204,  -0.33505857,  -1.63942921,  -2.81880975,  1.72370541,
      2.47747445,   -0.285959214, 1.51759779,   -2.30945373,  -1.98385429,
      -1.16285217,  -2.28210092,  -1.74110687,  -3.93382144,  0.272397488,
      -1.17788923,  9.38643741,   5.71728086,   6.73959827,   -0.794024169,
      2.3772893,    -2.78201818,  2.41477132,   1.20094347,   3.13179207};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkF32ReverseTest, kF32TestTypesTensorsWork3) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({3, 4, 5});
  const Shape shapeR({3, 4, 5});
  Vector<StorageT> operand_data{
      -1.15580153,  -1.18647683, 3.42077422,  -1.76529694,  -1.13649952,
      -2.85637474,  -5.04712057, 0.199800327, -3.37034297,  0.239038169,
      5.22281075,   -1.72277057, -2.76423025, -0.468372345, -0.401783675,
      1.29584312,   -2.68716073, -4.00673866, 3.02402735,   -3.562620e+00,
      -4.62835264,  0.933315098, -1.61677313, -3.28550196,  -5.13500834,
      5.07587385,   5.60044527,  3.78441501,  0.828205883,  -4.44870663,
      3.48453283,   7.26203108,  1.95305467,  -1.73211968,  -3.27272105,
      1.41651392,   1.80648303,  -3.19986081, 1.95118356,   1.22354436,
      3.70364332,   -1.52499723, -2.01245666, 1.88199496,   -5.90188169,
      -0.353008151, -4.29440308, 0.663977623, 0.137453571,  1.85718787,
      -2.92269301,  -5.28896284, 1.0011059,   1.41669655,   -7.62059211,
      -1.50831878,  -3.9411943,  0.403646946, -0.612915277, -0.300886393};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {2, 0, 1};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data{
      -0.300886393,  -0.612915277, 0.403646946, -3.9411943,  -1.50831878,
      -7.62059211,   1.41669655,   1.0011059,   -5.28896284, -2.92269301,
      1.85718787,    0.137453571,  0.663977623, -4.29440308, -0.353008151,
      -5.90188169,   1.88199496,   -2.01245666, -1.52499723, 3.70364332,
      1.22354436,    1.95118356,   -3.19986081, 1.80648303,  1.41651392,
      -3.27272105,   -1.73211968,  1.95305467,  7.26203108,  3.48453283,
      -4.44870663,   0.828205883,  3.78441501,  5.60044527,  5.07587385,
      -5.13500834,   -3.28550196,  -1.61677313, 0.933315098, -4.62835264,
      -3.562620e+00, 3.02402735,   -4.00673866, -2.68716073, 1.29584312,
      -0.401783675,  -0.468372345, -2.76423025, -1.72277057, 5.22281075,
      0.239038169,   -3.37034297,  0.199800327, -5.04712057, -2.85637474,
      -1.13649952,   -1.76529694,  3.42077422,  -1.18647683, -1.15580153};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkF32ReverseTest, kF32TestTypesTensorsWork4) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({3, 4, 5});
  const Shape shapeR({3, 4, 5});
  Vector<StorageT> operand_data{
      -0.464423895, -1.50158894, -2.0344243,    -1.41180933,   -2.90842533,
      0.823336541,  3.45140028,  3.69739723,    -1.89843452,   -1.68810058,
      3.40111327,   1.05722153,  2.11909556,    -0.850937306,  5.09096432,
      -1.23388743,  3.58495235,  -0.120762065,  2.2584517,     0.494466394,
      4.47528076,   0.917005598, 1.54741216,    -6.545910e+00, 2.31888652,
      -2.10922027,  -4.11765957, -5.467250e-01, 1.67117274,    4.70610762,
      -0.055694636, 1.16180658,  -8.108325,     5.54486752,    -4.16301775,
      -0.85471028,  0.89209491,  4.35790396,    8.51306534,    0.492473841,
      0.271108121,  6.0962081,   -1.66858292,   2.68769217,    2.20100784,
      -5.08213377,  1.03311872,  2.17707014,    -3.29460859,   -1.56388259,
      -1.33692706,  2.77195072,  0.144049063,   5.18237638,    5.3109436,
      1.95865726,   6.17927361,  1.1321938,     1.05025399,    1.04188859};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data{
      -0.464423895, -1.50158894, -2.0344243,    -1.41180933,   -2.90842533,
      0.823336541,  3.45140028,  3.69739723,    -1.89843452,   -1.68810058,
      3.40111327,   1.05722153,  2.11909556,    -0.850937306,  5.09096432,
      -1.23388743,  3.58495235,  -0.120762065,  2.2584517,     0.494466394,
      4.47528076,   0.917005598, 1.54741216,    -6.545910e+00, 2.31888652,
      -2.10922027,  -4.11765957, -5.467250e-01, 1.67117274,    4.70610762,
      -0.055694636, 1.16180658,  -8.108325,     5.54486752,    -4.16301775,
      -0.85471028,  0.89209491,  4.35790396,    8.51306534,    0.492473841,
      0.271108121,  6.0962081,   -1.66858292,   2.68769217,    2.20100784,
      -5.08213377,  1.03311872,  2.17707014,    -3.29460859,   -1.56388259,
      -1.33692706,  2.77195072,  0.144049063,   5.18237638,    5.3109436,
      1.95865726,   6.17927361,  1.1321938,     1.05025399,    1.04188859};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

TYPED_TEST(NonQuantizedkF32ReverseTest, kF32TestTypesTensorsWork5) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({4, 5});
  const Shape shapeR({4, 5});
  Vector<StorageT> operand_data{
      -6.02407265,  3.11093283,  -2.532550e+00, -4.78078699, 2.53503942,
      1.270130e+00, -1.59578395, 3.28086519,    -1.46065211, -4.57731438,
      -0.32994166,  3.34717584,  2.85873795,    -3.4305625,  1.44537222,
      -0.922935426, -4.55431795, 6.62983989,    3.72943854,  5.40398884};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data{
      -0.922935426, -4.55431795, 6.62983989,    3.72943854,  5.40398884,
      -0.32994166,  3.34717584,  2.85873795,    -3.4305625,  1.44537222,
      1.270130e+00, -1.59578395, 3.28086519,    -1.46065211, -4.57731438,
      -6.02407265,  3.11093283,  -2.532550e+00, -4.78078699, 2.53503942};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

using kBF16TestTypes = ::testing::Types<TestParam<DataType::kBF16>>;
template <class T>
struct NonQuantizedkBF16ReverseTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkBF16ReverseTest, kBF16TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkBF16ReverseTest, kBF16TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({4, 5});
  const Shape shapeR({4, 5});
  Vector<__bf16> operand_data{
      -2.843750e+00, -4.281250e+00, 2.937500e+00,  -5.000000e+00, -1.101560e+00,
      1.765630e+00,  7.812500e-01,  -2.984380e+00, 2.312500e+00,  -4.453130e-01,
      -6.484380e-01, 1.929690e+00,  3.964840e-01,  1.320310e+00,  -2.343750e+00,
      -1.656250e+00, 5.742190e-01,  -1.179690e+00, -4.750000e+00, 7.617180e-02};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0};
  Vector<__bf16> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<__bf16> expected_data{
      -1.656250e+00, 5.742190e-01,  -1.179690e+00, -4.750000e+00,
      7.617180e-02,  -6.484380e-01, 1.929690e+00,  3.964840e-01,
      1.320310e+00,  -2.343750e+00, 1.765630e+00,  7.812500e-01,
      -2.984380e+00, 2.312500e+00,  -4.453130e-01, -2.843750e+00,
      -4.281250e+00, 2.937500e+00,  -5.000000e+00, -1.101560e+00};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

using kF16TestTypes = ::testing::Types<TestParam<DataType::kF16>>;
template <class T>
struct NonQuantizedkF16ReverseTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedkF16ReverseTest, kF16TestTypes, TestParamNames);

TYPED_TEST(NonQuantizedkF16ReverseTest, kBF16TestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({4, 5});
  const Shape shapeR({4, 5});
  Vector<StorageT> operand_data{
      -3.843750e+00, -1.816410e+00, -3.042970e+00, 3.191410e+00, 9.692380e-01,
      -4.926760e-01, -4.389650e-01, -3.982420e+00, 2.021480e+00, -2.193360e+00,
      2.759770e+00,  -1.366210e+00, -1.299800e+00, 1.706050e+00, 1.254880e+00,
      -2.023930e-01, -1.726560e+00, 3.976560e+00,  1.223630e+00, -3.080080e+00};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data{
      -2.023930e-01, -1.726560e+00, 3.976560e+00,  1.223630e+00, -3.080080e+00,
      2.759770e+00,  -1.366210e+00, -1.299800e+00, 1.706050e+00, 1.254880e+00,
      -4.926760e-01, -4.389650e-01, -3.982420e+00, 2.021480e+00, -2.193360e+00,
      -3.843750e+00, -1.816410e+00, -3.042970e+00, 3.191410e+00, 9.692380e-01};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));

  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}

template <class T>
struct NonQuantizedBoolDotGeneralTest : ::testing::Test {};
TYPED_TEST_SUITE(NonQuantizedBoolDotGeneralTest, BoolTestType, TestParamNames);

TYPED_TEST(NonQuantizedBoolDotGeneralTest, BoolTestTypesTensorsWork1) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shapeOperand({4, 5});
  const Shape shapeR({4, 5});
  Vector<StorageT> operand_data{true, true, true, true, true, true, true,
                                true, true, true, true, true, true, true,
                                true, true, true, true, true, true};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data{true, true, true, true, true, true, true,
                                 true, true, true, true, true, true, true,
                                 true, true, true, true, true, true};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  ASSERT_OK(Prepare(op, operand, output_tensor));
  ASSERT_OK(Evaluate(op, operand, output_tensor));
  EXPECT_THAT(output_data, Pointwise(FloatEq(), expected_data));
}
// Negative test case for constraint 1
using QuantizedNegativeTestTypes =
    ::testing::Types<TestParam<DataType::kSI4, DataType::kBF16>,
                     TestParam<DataType::kSI8, DataType::kBF16>,
                     TestParam<DataType::kSI4, DataType::kF16>,
                     TestParam<DataType::kSI8, DataType::kF16>>;

template <class T>
struct QuantizedIntReverseTestNeg : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedIntReverseTestNeg, QuantizedNegativeTestTypes,
                 TestParamNames);

TYPED_TEST(QuantizedIntReverseTestNeg, QuantizedTestTypesRaiseAnError1) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;
  const Shape shapeOperand({3, 2});
  const Shape shapeR({3, 2});

  Vector<int32_t> operand_data_int =
      Vector<int32_t>{(1), (2), (3), (4), (5), (6)};
  Vector<StorageT> operand_data(operand_data_int.begin(),
                                operand_data_int.end());
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0, 1};
  Vector<StorageT> output_data(shapeR.NumElements());
  const ExpressedT scale = static_cast<ExpressedT>(2);
  const StorageT zero_point = static_cast<StorageT>(0);
  const ExpressedT scale_sv = static_cast<ExpressedT>(1.2);
  const StorageT zero_point_sv = static_cast<StorageT>(1);
  const QuantizedElementTypePerTensor tensor_type1 =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point,
                                    TypeParam::kExpressed, scale);
  const QuantizedElementTypePerTensor tensor_type2 =
      QuantizedElementTypePerTensor(TypeParam::kStorage, zero_point_sv,
                                    DataType::kF32, scale_sv);

  Tensor operand{
      .type = QuantizedPerTensorTensorType{.shape = shapeOperand,
                                           .element_type = tensor_type1},
      .data = operand_data.data()};
  Tensor output{
      .type = QuantizedPerTensorTensorType{.shape = shapeR,
                                           .element_type = tensor_type2},
      .data = output_data.data()};
  auto op = Create(reverseOp::Attributes{.dimensions = dimensions});

  Vector<float> expected_data = Vector<float>{(6), (5), (4), (3), (2), (1)};

  const absl::Status status = Prepare(op, operand, output);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(),
              ("stablehlo.reverse: baseline type constraint is not satisfied " +
               std::visit([](auto v) -> std::string { return ToString(v); },
                          operand.element_type()) +
               " and " +
               std::visit([](auto v) -> std::string { return ToString(v); },
                          output.element_type()) +
               "."));
}

TYPED_TEST(NonQuantizedIntReverseTest, IntTestTypesTensorsRaiseAnError2) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({3, 2});
  const Shape shapeR({3, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{(1), (2), (3), (4), (5), (6)};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {0, 1};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = DataType::kBF16},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data =
      Vector<StorageT>{(6), (5), (4), (3), (2), (1)};

  Tensor expected_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = expected_data.data()};

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(),
              "The element type of operand and result must be same.");
}

// Negative test case for constraint 2
TYPED_TEST(NonQuantizedIntReverseTest, IntTestTypesRaiseAnError3) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({3, 2});
  const Shape shapeR({3, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{(1), (2), (3), (4), (5), (6)};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {1, 1};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data =
      Vector<StorageT>{(6), (5), (4), (3), (2), (1)};

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(),
              "Every dimension to be reversed must be unique.");
}

TYPED_TEST(NonQuantizedIntReverseTest, IntTestTypesRaiseAnError4) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({3, 2});
  const Shape shapeR({3, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{(1), (2), (3), (4), (5), (6)};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {7, 1};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data =
      Vector<StorageT>{(6), (5), (4), (3), (2), (1)};

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(), "Dimension out of range.");
}

// Negative test case for constraint 3
TYPED_TEST(NonQuantizedIntReverseTest, IntTestTypesRaiseAnError5) {
  using StorageT = typename TypeParam::StorageT;
  const Shape shapeOperand({3, 2});
  const Shape shapeR({3, 2});
  Vector<StorageT> operand_data =
      Vector<StorageT>{(1), (2), (3), (4), (5), (6)};
  absl::InlinedVector<Axis, kMaxNumDimensions> dimensions = {5, 1};
  Vector<StorageT> output_data(shapeR.NumElements());

  Tensor operand{.type = TensorType{.shape = shapeOperand,
                                    .element_type = TypeParam::kStorage},
                 .data = operand_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shapeR, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(reverseOp::Attributes{
      .dimensions = dimensions,
  });

  Vector<StorageT> expected_data =
      Vector<StorageT>{(6), (5), (4), (3), (2), (1)};

  const absl::Status status = Prepare(op, operand, output_tensor);
  EXPECT_THAT(status, shlo_ref::testing::StatusIs(
                          absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(status.message(),
              "The dimensions must be in the range of the rank of result.");
}

}  // namespace
}  // namespace shlo_ref