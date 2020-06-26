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
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

using namespace mlir::rise;

namespace mlir {
namespace edsc {
// arrayT
namespace type {
FunType funtype(Type in, Type out);
Nat nat(int val);
ArrayType array(Nat size, DataType elemType);
ArrayType array(int size, DataType elemType);
ArrayType array2D(Nat outerSize, Nat innerSize, DataType elemType);
ArrayType array2D(int outerSize, int innerSize, DataType elemType);
ArrayType array3D(Nat outerSize, Nat midSize, Nat innerSize, DataType elemType);
ArrayType array3D(int outerSize, int midSize, int innerSize, DataType elemType);
Tuple tuple(DataType lhs, DataType rhs);
ScalarType scalar(Type wrappedType);
ScalarType scalarF32();
ScalarType scalarF64();
} // namespace type

namespace abstraction {
Value sumLambda(ScalarType summandType);
Value multAndSumUpLambda(ScalarType summandType);
Value slide2d(Nat szOuter, Nat stOuter, Nat szInner, Nat stInner,
              Value array2DVal);

} // namespace abstraction

namespace op {
using lambdaBodyBuilder = function_ref<Value(MutableArrayRef<BlockArgument>)>;

// frontend edsc
Value padClamp(Nat l, Nat r, Value array);
Value slide(Nat sz, Nat sp, Value array);
Value fst(Value tuple);
Value snd(Value tuple);
Value zip(Value lhs, Value rhs);
Value transpose(Value array);
Value mapSeq(StringRef lowerTo, DataType s, DataType t, Value lambda,
             Value array);
Value mapSeq(StringRef lowerTo, DataType s, DataType t,
             lambdaBodyBuilder bodyBuilder,
             Value array);
//Value reduceSeq(Value lambda, Value initializer, Value array);
Value reduceSeq(StringRef lowerTo, Value lambda, Value initializer,
                Value array);
Value reduceSeq(StringRef lowerTo, DataType s, DataType t, lambdaBodyBuilder bodyBuilder, Value initializer,
                Value array);

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
Value padClamp(Nat n, Nat l, Nat r, DataType t);
Value padClamp(Nat n, Nat l, Nat r, DataType t, Value array);
Value lambda(FunType lambdaType,
             lambdaBodyBuilder bodyBuilder);
// TODO: Lambda with named args
//Value mlir::edsc::op::lambda(
//    FunType lambdaType,
//    function_ref<Value(BlockArgument, BlockArgument)> bodyBuilder);
//
//Value mlir::edsc::op::lambda(
//    FunType lambdaType,
//    function_ref<Value(BlockArgument, BlockArgument, BlockArgument)> bodyBuilder);
//
//Value mlir::edsc::op::lambda(
//    FunType lambdaType,
//    function_ref<Value(BlockArgument, BlockArgument, BlockArgument, BlockArgument)> bodyBuilder);
// Value embed(Type result, ValueRange exposedValues,
// function_ref<void(MutableArrayRef<BlockArgument>)> bodyBuilder);
Value embed(Type result, ValueRange exposedValues,
            function_ref<Value(MutableArrayRef<BlockArgument>)> bodyBuilder);
Value mapSeq(StringRef lowerTo, Nat n, DataType s, DataType t);
Value mapSeq(StringRef lowerTo, Nat n, DataType s, DataType t, Value lambda,
             Value array);
Value reduceSeq(StringRef lowerTo, Nat n, DataType s, DataType t);
Value reduceSeq(StringRef lowerTo, Nat n, DataType s, DataType t, Value lambda,
                Value initializer, Value array);

void rise_return(Value returnValue);
void out(Value writeTo, Value result);

} // namespace op

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_RISE_EDSC_BUILDERS_H_
