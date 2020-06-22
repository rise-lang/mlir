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

#ifndef MLIR_DIALECT_RISE_EDSC_BUILDERS_H_
#define MLIR_DIALECT_RISE_EDSC_BUILDERS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

using namespace mlir::rise;

namespace mlir {
namespace edsc {

// TODO: something like Lambdabuilder, which can nest the creationcode for inside
//class AffineLoopNestBuilder {
//public:
//  /// This entry point accommodates the fact that AffineForOp implicitly uses
//  /// multiple `lbs` and `ubs` with one single `iv` and `step` to encode `max`
//  /// and and `min` constraints respectively.
//  AffineLoopNestBuilder(Value *iv, ArrayRef<Value> lbs, ArrayRef<Value> ubs,
//                        int64_t step);
//  AffineLoopNestBuilder(MutableArrayRef<Value> ivs, ArrayRef<Value> lbs,
//                        ArrayRef<Value> ubs, ArrayRef<int64_t> steps);
//
//  void operator()(function_ref<void(void)> fun = nullptr);
//
//private:
//  SmallVector<LoopBuilder, 4> loops;
//};

//class LambdaBuilder :


namespace op {


Value in(Value in, Type type);
Value literal(DataType t, StringRef literal);
Value zip(Nat n, DataType s, DataType t);
Value zip(Nat n, DataType s, DataType t, Value lhs, Value rhs);
Value fst(DataType s, DataType t);
Value fst(DataType s, DataType t, Value tuple);
Value snd(DataType s, DataType t);
Value snd(DataType s, DataType t, Value tuple);
Value transpose(Nat n, Nat m, DataType t);
Value transpose(Nat n, Nat m, DataType t, Value array);
Value slide(Nat n, Nat sz, Nat sp, DataType t);
Value slide(Nat n, Nat sz, Nat sp, DataType t, Value array);
Value lambda(FunType lambdaType, function_ref<Value(MutableArrayRef<BlockArgument>)> bodyBuilder);
Value embed(Type result, ValueRange exposedValues, function_ref<void(MutableArrayRef<BlockArgument>)> bodyBuilder);
Value mapSeq(StringRef lowerTo, Nat n, DataType s, DataType t);
Value mapSeq(StringRef lowerTo, Nat n, DataType s, DataType t, Value lambda, Value array);
Value reduceSeq(StringRef lowerTo, Nat n, DataType s, DataType t);
Value reduceSeq(StringRef lowerTo, Nat n, DataType s, DataType t, Value lambda, Value initializer, Value array);

void rise_return(Value returnValue);
void out(Value writeTo, Value result);

} // namespace op


} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_RISE_EDSC_BUILDERS_H_
