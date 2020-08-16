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

#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

using namespace mlir::rise;

namespace mlir {
namespace edsc {
namespace type {

FunType funType(Type in, Type out);
Nat natType(int val);
ArrayType arrayType(Nat size, DataType elemType);
ArrayType arrayType(int size, DataType elemType);
ArrayType array2DType(Nat outerSize, Nat innerSize, DataType elemType);
ArrayType array2DType(int outerSize, int innerSize, DataType elemType);
ArrayType array3DType(Nat outerSize, Nat midSize, Nat innerSize,
                      DataType elemType);
ArrayType array3DType(int outerSize, int midSize, int innerSize,
                      DataType elemType);
Tuple tupleType(DataType lhs, DataType rhs);
ScalarType scalarType(Type wrappedType);
ScalarType scalarF32Type();
ScalarType scalarF64Type();
} // namespace type

namespace abstraction {
Value mapSeq2D(DataType resultElemType,
               function_ref<Value(BlockArgument)> bodyBuilder, Value array2D);
Value mapSeq2D(StringRef lowerTo, DataType resultElemType,
               function_ref<Value(BlockArgument)> bodyBuilder, Value array2D);
Value slide2D(Nat szOuter, Nat stOuter, Nat szInner, Nat stInner,
              Value array2DVal);
Value pad2D(Nat lOuter, Nat rOuter, Nat lInner, Nat rInner, Value array);
Value zip2D(Value array2DA, Value array2DB);

Value sumLambda(ScalarType summandType);          // return val1 + val2
Value multAndSumUpLambda(ScalarType summandType); // return (val1 + val2) * val3

} // namespace abstraction

namespace op {
using lambdaBodyBuilder = function_ref<Value(MutableArrayRef<BlockArgument>)>;

//===----------------------------------------------------------------------===//
// Rise Frontend EDSC
//===----------------------------------------------------------------------===//

// Core Lambda-Calculus
Value lambda(FunType lambdaType, lambdaBodyBuilder bodyBuilder);
Value apply(DataType resultType, Value fun, ValueRange args);

// Interoperability
Value in(Value in, Type type);
void out(Value writeTo, Value result);
Value embed(Type result, ValueRange exposedValues,
            function_ref<Value(MutableArrayRef<BlockArgument>)> bodyBuilder);
void rise_return(Value returnValue);
void lowering_unit(function_ref<void()> bodyBuilder);

// Patterns
Value mapSeq(function_ref<Value(BlockArgument)> bodyBuilder, Value array);
Value mapSeq(DataType resultElemType,
             function_ref<Value(BlockArgument)> bodyBuilder, Value array);
Value mapSeq(StringRef lowerTo, DataType resultElemType,
             function_ref<Value(BlockArgument)> bodyBuilder, Value array);
Value map(DataType resultElemType,
          function_ref<Value(BlockArgument)> bodyBuilder, Value array);
Value reduceSeq(DataType resulType,
                function_ref<Value(BlockArgument, BlockArgument)> bodyBuilder,
                Value initializer, Value array);
Value zip(Value lhs, Value rhs);
// Value tuple(Value lhs, Value rhs);
Value fst(Value tuple);
Value snd(Value tuple);
Value split(Nat n, Value array);
Value join(Value array);
Value transpose(Value array);
Value slide(Nat windowSize, Nat step, Value array);
Value padClamp(Nat l, Nat r, Value array);
Value literal(DataType t, StringRef literal);

//===----------------------------------------------------------------------===//
// Rise other convenience EDSC
//===----------------------------------------------------------------------===//

// clang-format off
Value lambda(ArrayRef<Type> types, function_ref<Value(BlockArgument)> bodyBuilder);
Value lambda1(FunType lambdaType,
              function_ref<Value(BlockArgument)> bodyBuilder);
Value lambda2(FunType lambdaType,
              function_ref<Value(BlockArgument, BlockArgument)> bodyBuilder);
Value lambda3(FunType lambdaType,
              function_ref<Value(BlockArgument, BlockArgument, BlockArgument)> bodyBuilder);
Value lambda4(FunType lambdaType,
              function_ref<Value(BlockArgument, BlockArgument, BlockArgument, BlockArgument)> bodyBuilder);
Value lambda5(FunType lambdaType,
              function_ref<Value(BlockArgument, BlockArgument, BlockArgument, BlockArgument, BlockArgument)> bodyBuilder);
Value lambda6(FunType lambdaType,
              function_ref<Value(BlockArgument, BlockArgument, BlockArgument, BlockArgument, BlockArgument, BlockArgument)> bodyBuilder);
Value embed1(Type result, ValueRange exposedValues,
             function_ref<Value(BlockArgument)> bodyBuilder);
Value embed2(Type result, ValueRange exposedValues,
             function_ref<Value(BlockArgument, BlockArgument)> bodyBuilder);
Value embed3(Type result, ValueRange exposedValues,
             function_ref<Value(BlockArgument, BlockArgument, BlockArgument)> bodyBuilder);
Value embed4(Type result, ValueRange exposedValues,
             function_ref<Value(BlockArgument, BlockArgument, BlockArgument, BlockArgument)> bodyBuilder);
Value embed5(Type result, ValueRange exposedValues,
             function_ref<Value(BlockArgument, BlockArgument, BlockArgument, BlockArgument, BlockArgument)> bodyBuilder);
Value embed6(Type result, ValueRange exposedValues,
             function_ref<Value(BlockArgument, BlockArgument, BlockArgument, BlockArgument, BlockArgument, BlockArgument)> bodyBuilder);
// clang-format on
Value mapSeq(StringRef lowerTo, DataType s, DataType t, Value lambda,
             Value array);
Value mapSeq(StringRef lowerTo, DataType s, DataType t,
             function_ref<Value(BlockArgument)> bodyBuilder, Value array);
Value mapSeq(StringRef lowerTo, DataType t,
             function_ref<Value(BlockArgument)> bodyBuilder, Value array);
Value mapSeq(StringRef lowerTo, Nat n, DataType s, DataType t);
Value mapSeq(StringRef lowerTo, Nat n, DataType s, DataType t, Value lambda,
             Value array);
Value mapSeq(StringRef lowerTo, DataType s, DataType t, Value lambda,
             Value array);
Value map(DataType t, Value lambda, Value array);
Value map(Nat n, DataType s, DataType t, Value lambda);
Value map(Nat n, DataType s, DataType t, Value lambda, Value array);
Value reduceSeq(StringRef lowerTo, DataType t,
                function_ref<Value(BlockArgument, BlockArgument)> bodyBuilder,
                Value initializer, Value array);
Value reduceSeq(StringRef lowerTo, DataType t, Value lambda, Value initializer,
                Value array);
Value reduceSeq(DataType t, Value lambda, Value initializer, Value array);
Value reduceSeq(StringRef lowerTo, Nat n, DataType s, DataType t);
Value reduceSeq(StringRef lowerTo, Nat n, DataType s, DataType t, Value lambda,
                Value initializer, Value array);
Value zip(Nat n, DataType s, DataType t);
Value zip(Nat n, DataType s, DataType t, Value lhs, Value rhs);
Value fst(DataType s, DataType t);
Value fst(DataType s, DataType t, Value tuple);
Value snd(DataType s, DataType t);
Value snd(DataType s, DataType t, Value tuple);
Value split(Nat n, Nat m, DataType t);
Value split(Nat n, Nat m, DataType t, Value array);
Value join(Nat n, Nat m, DataType t);
Value join(Nat n, Nat m, DataType t, Value array);
Value transpose(Nat n, Nat m, DataType t);
Value transpose(Nat n, Nat m, DataType t, Value array);
Value slide(Nat n, Nat sz, Nat sp, DataType t);
Value slide(Nat n, Nat sz, Nat sp, DataType t, Value array);
Value padClamp(Nat n, Nat l, Nat r, DataType t);
Value padClamp(Nat n, Nat l, Nat r, DataType t, Value array);

} // namespace op

} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_RISE_EDSC_BUILDERS_H_
