//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Rise/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

//using namespace mlir::rise;

namespace mlir {
namespace edsc {

// I have to give the minimal args to this and call the Valuebuilder templated
// with the Operation. It calls the builder of it with the args I provide here.

Value orr(Value lhs, Value rhs) {
  assert(lhs.getType().isInteger(1) && "expected boolean expression on LHS");
  assert(rhs.getType().isInteger(1) && "expected boolean expression on RHS");
  return ValueBuilder<OrOp>(lhs, rhs);
}

Value in(Value in, Type type) {
  assert(in.getType().isa<MemRefType>());
  return ValueBuilder<rise::InOp>(type, in);
}

Value mlir::edsc::op::mapSeq(StringRef lowerTo, rise::Nat n, rise::DataType s, rise::DataType t) {
// MapB
  rise::FunType mapType = rise::FunType::get(
      ScopedContext::getContext(),
      rise::FunType::get(ScopedContext::getContext(), s, t),
      rise::FunType::get(
          ScopedContext::getContext(),
          rise::ArrayType::get(ScopedContext::getContext(), n, s),
  rise::ArrayType::get(ScopedContext::getContext(), n, t)));

  return ValueBuilder<rise::MapSeqOp>(mapType, rise::NatAttr::get(ScopedContext::getContext(), n),
                                      rise::DataTypeAttr::get(ScopedContext::getContext(), s),
                                      rise::DataTypeAttr::get(ScopedContext::getContext(), t),
                                      StringAttr::get(lowerTo, ScopedContext::getContext()));
}

//Value mlir::edsc::op::mapSeq(StringRef lowerTo, rise::Nat n, rise::DataType s, rise::DataType t, Value lambda, Value array) {
//  Value mapSeq = mlir::edsc::op::mapSeq(lowerTo, n, s, t);
//
//  return ValueBuilder<rise::ApplyOp>(
//      rise::ArrayType::get(ScopedContext::getContext(), n, s),
//      mapSeq,
//      ValueRange{lambda, array});
//}

//Value mapSeq(StringRef lowerTo, rise::Nat n, rise::DataType s, rise::DataType t, function_ref<Value(void)> createLambda, Value array) {
//
//  auto lambda = createLambda();
//  Value mapSeq = mlir::edsc::op::mapSeq(n,s,t,lowerTo);
//
//  return ValueBuilder<rise::ApplyOp>(
//      rise::ArrayType::get(ScopedContext::getContext(), n, s),
//      mapSeq,
//      ValueRange{lambda, array});
//}


} // namespace edsc
} // namespace mlir
