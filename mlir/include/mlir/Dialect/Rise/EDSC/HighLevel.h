//===- Builders.h - MLIR Declarative Builder Classes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides intuitive composable interfaces for building structured MLIR
// snippets in a declarative fashion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_RISE_EDSC_HIGHLEVEL_H_
#define MLIR_DIALECT_RISE_EDSC_HIGHLEVEL_H_

#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Rise/EDSC/Builders.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/invoke.hpp"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include <iostream>

using namespace mlir::rise;
using namespace mlir::edsc;
using namespace mlir::edsc::type;
using namespace mlir::edsc::op;
using namespace mlir::edsc::intrinsics;

namespace mlir {
namespace {
Type inferTypeForInOp(Value input) {
  if (MemRefType inMemrefType = input.getType().dyn_cast<MemRefType>()) {
    // infer input type from memref
    DataType inputType = scalarF32Type();
    for (auto size = inMemrefType.getShape().rbegin();
         size != inMemrefType.getShape().rend(); size++) {
      inputType = arrayType(*size, inputType);
    }
    return inputType;
  }
  emitError(input.getLoc())
      << "Could not infer type for rise.in operation with provided input!";
  return nullptr;
}

template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each(std::tuple<Tp...> &, FuncT) {}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type for_each(std::tuple<Tp...> &t, FuncT f) {
  f(std::get<I>(t));
  for_each<I + 1, FuncT, Tp...>(t, f);
}

template <typename T, typename... Targs>
struct builder {
  Value output;
  std::tuple<T, Targs...> inputs;
  builder(Value output, std::tuple<T, Targs...> inputs)
      : output{output}, inputs{inputs} {}

  void operator()(std::function<Value(T, Targs...)> fun) {
    lowering_unit([&]() {
      std::array<T, std::tuple_size<std::tuple<T, Targs...>>::value> ins;
      int index = 0;
      for_each(inputs, [&](Value elem) {
        ins[index++] = (in(elem, inferTypeForInOp(elem)));
      });
      Value result = invoke_hpp::apply(fun, ins);
      out(output, result);
    });
  }
};
} // namespace

namespace edsc {
namespace highlevel {

template <typename T, typename... Targs>
builder<T, Targs...> makeRiseProgram(Value output, T input, Targs... inputs) {
  return builder<T, Targs...>{output, std::make_tuple(input, inputs...)};
}

Value matrix_multiplication(int M, int N, int K, Value A, Value B);
Value conv2D(Value input, Value kernel);
Value conv2D(Value input, Value kernel, int padl, int padr, int padt, int padb);
Value conv2DSeparated(Value input, Value kernelH, Value kernelV, int padl,
                      int padr, int padt, int padb);
Value conv2DTF(Value input, Value kernel);
Value stencil(int N, int windowSize, int step, Value input);
Value stencil2D(int M, int N, int outerWindowSize, int outerStep,
                int innerWindowSize, int innerStep, Value input);

// utilities
void generateTest(int dims, ArrayRef<int64_t> inSizes,
                  ArrayRef<int64_t> outSizes, FuncOp riseFun = nullptr);
void generateTest(int dims, ArrayRef<int64_t> inSizesA,
                  ArrayRef<int64_t> inSizesB, ArrayRef<int64_t> outSizes,
                  FuncOp riseFun = nullptr);
void generateTest(int dims, ArrayRef<int64_t> inSizesA,
                  ArrayRef<int64_t> inSizesB, ArrayRef<int64_t> inSizesC,
                  ArrayRef<int64_t> outSizes, FuncOp riseFun);

} // namespace highlevel

namespace utils {
Value getFilledMemRef(ArrayRef<int64_t> shape, float fillValue,
                      Value memref = nullptr);
Value getFilledMemRef(ArrayRef<int64_t> shape, Value memref = nullptr);
template <typename T, typename... Targs>
void makeRiseTest(FuncOp riseFun,
                                      ArrayRef<int64_t> outputShape, T input,
                                      Targs... inputs) {
  Value output = std_alloc(MemRefType::get(outputShape, FloatType::getF32(ScopedContext::getContext()), {}, 0));
  std::tuple<T, Targs...> tuple = std::make_tuple(input, inputs...);

  Value cst0 = std_constant_index(0);
  Value step = std_constant_index(1);
  Value repetitions = std_constant_index(300);
  loopNestBuilder(cst0, repetitions, step, [&](auto ivs) {
    Value t0 = std_call("rtclock", ArrayRef<Type>{FloatType::getF64(ScopedContext::getContext())}).op.getResult(0);
    std_call(riseFun, ValueRange{input, inputs..., output});
    Value t1 = std_call("rtclock", ArrayRef<Type>{FloatType::getF64(ScopedContext::getContext())}).op.getResult(0);
    std_call("print_time", ArrayRef<Type>(), ValueRange{t0, t1});
  });

  for_each(tuple, [&](Value elem){
    Value casted = std_memref_cast(elem, UnrankedMemRefType::get(FloatType::getF32(ScopedContext::getContext()), 0));
    std_call("print_memref_f32", ArrayRef<Type>(), ValueRange{casted});
  });
  Value casted = std_memref_cast(output, UnrankedMemRefType::get(FloatType::getF32(ScopedContext::getContext()), 0));
  std_call("print_memref_f32", ArrayRef<Type>(), ValueRange{casted});
}
} // namespace utils
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_RISE_EDSC_HIGHLEVEL_H_
