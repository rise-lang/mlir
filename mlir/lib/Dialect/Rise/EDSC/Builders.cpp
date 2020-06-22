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

using namespace mlir::rise;

namespace mlir {
namespace edsc {

Value mlir::edsc::op::in(Value in, Type type) {
  assert(in.getType().isa<MemRefType>());
  return ValueBuilder<InOp>(type, in);
}

Value mlir::edsc::op::literal(DataType t, StringRef literal) {
  MLIRContext *context = ScopedContext::getContext();
  return ValueBuilder<LiteralOp>(t,
                                 LiteralAttr::get(context, t, literal.str()));
}

Value mlir::edsc::op::reduceSeq(StringRef lowerTo, Nat n, DataType s,
                                DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  FunType reduceType = FunType::get(
      context, FunType::get(context, s, FunType::get(context, t, t)),
      FunType::get(context, t,
                   FunType::get(context, ArrayType::get(context, n, s), t)));
  return ValueBuilder<ReduceSeqOp>(
      reduceType, NatAttr::get(context, n), DataTypeAttr::get(context, s),
      DataTypeAttr::get(context, t), StringAttr::get(lowerTo, context));
}

Value mlir::edsc::op::reduceSeq(StringRef lowerTo, Nat n, DataType s,
                                DataType t, Value lambda, Value initializer,
                                Value array) {
  Value reduceOp = reduceSeq(lowerTo, n, s, t);
  return ValueBuilder<ApplyOp>(t, reduceOp,
                               ValueRange{lambda, initializer, array});
}

Value mlir::edsc::op::zip(Nat n, DataType s, DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  FunType zipType = FunType::get(
      context, ArrayType::get(context, n, s),
      FunType::get(context, ArrayType::get(context, n, t),
                   ArrayType::get(context, n, Tuple::get(context, s, t))));
  return ValueBuilder<ZipOp>(zipType, NatAttr::get(context, n),
                             DataTypeAttr::get(context, s),
                             DataTypeAttr::get(context, t));
}

Value mlir::edsc::op::zip(Nat n, DataType s, DataType t, Value lhs, Value rhs) {
  MLIRContext *context = ScopedContext::getContext();

  Value zipOp = zip(n, s, t);
  return ValueBuilder<ApplyOp>(
      ArrayType::get(context, n, Tuple::get(context, s, t)), zipOp,
      ValueRange{lhs, rhs});
}

Value mlir::edsc::op::fst(DataType s, DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  return ValueBuilder<FstOp>(
      FunType::get(context, Tuple::get(context, s, t), s),
      DataTypeAttr::get(context, s), DataTypeAttr::get(context, t));
}

Value mlir::edsc::op::fst(DataType s, DataType t, Value tuple) {
  Value fstOp = fst(s, t);
  return ValueBuilder<ApplyOp>(s, fstOp, tuple);
}

Value mlir::edsc::op::snd(DataType s, DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  return ValueBuilder<SndOp>(
      FunType::get(context, Tuple::get(context, s, t), t),
      DataTypeAttr::get(context, s), DataTypeAttr::get(context, t));
}

Value mlir::edsc::op::snd(DataType s, DataType t, Value tuple) {
  Value sndOp = snd(s, t);
  return ValueBuilder<ApplyOp>(t, sndOp, tuple);
}

Value mlir::edsc::op::transpose(Nat n, Nat m, DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  FunType transposeType = FunType::get(
      context, ArrayType::get(context, n, ArrayType::get(context, m, t)),
      ArrayType::get(context, m, ArrayType::get(context, n, t)));

  return ValueBuilder<TransposeOp>(transposeType, NatAttr::get(context, n),
                                   NatAttr::get(context, m),
                                   DataTypeAttr::get(context, t));
}

Value mlir::edsc::op::transpose(Nat n, Nat m, DataType t, Value array) {
  MLIRContext *context = ScopedContext::getContext();
  Value transposeOp = transpose(n, m, t);
  return ValueBuilder<ApplyOp>(
      ArrayType::get(context, m, ArrayType::get(context, n, t)), transposeOp,
      ValueRange{array});
}

Value mlir::edsc::op::slide(Nat n, Nat sz, Nat sp, DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  FunType slideType = FunType::get(
      context,
      ArrayType::get(context,
                     Nat::get(context, sp.getIntValue() * n.getIntValue() +
                                           sz.getIntValue() - sp.getIntValue()),
                     t),
      ArrayType::get(context, n, ArrayType::get(context, sz, t)));

  return ValueBuilder<SlideOp>(
      slideType, NatAttr::get(context, n), NatAttr::get(context, sz),
      NatAttr::get(context, sp), DataTypeAttr::get(context, t));
}

Value mlir::edsc::op::slide(Nat n, Nat sz, Nat sp, DataType t, Value array) {
  MLIRContext *context = ScopedContext::getContext();
  Value slideOp = slide(n, sz, sp, t);
  return ValueBuilder<ApplyOp>(
      ArrayType::get(context, n, ArrayType::get(context, sz, t)), slideOp,
      ValueRange{array});
}

Value mlir::edsc::op::mapSeq(StringRef lowerTo, Nat n, DataType s, DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  FunType mapType =
      FunType::get(context, FunType::get(context, s, t),
                   FunType::get(context, ArrayType::get(context, n, s),
                                ArrayType::get(context, n, t)));

  return ValueBuilder<MapSeqOp>(
      mapType, NatAttr::get(context, n), DataTypeAttr::get(context, s),
      DataTypeAttr::get(context, t), StringAttr::get(lowerTo, context));
}

Value mlir::edsc::op::mapSeq(StringRef lowerTo, Nat n, DataType s, DataType t,
                             Value lambda, Value array) {
  MLIRContext *context = ScopedContext::getContext();
  Value mapSeq = mlir::edsc::op::mapSeq(lowerTo, n, s, t);
  return ValueBuilder<ApplyOp>(ArrayType::get(context, n, s), mapSeq,
                               ValueRange{lambda, array});
}

// TODO: maybe change the builder to automatically introduce a return for the
// last Value created in bodybuilder - replace void with Value
Value mlir::edsc::op::lambda(
    FunType lambdaType,
    function_ref<Value(MutableArrayRef<BlockArgument>)> bodyBuilder) {
  return ValueBuilder<LambdaOp>(
      lambdaType, [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      MutableArrayRef<BlockArgument> args) {
          ScopedContext nestedContext(nestedBuilder, nestedLoc);
          OpBuilder::InsertionGuard guard(nestedBuilder);
          return bodyBuilder(args);
      });
}

// TODO maybe a builder like:
// I mean something that we can access blockargs better, not 100% sure how
// Value mlir::edsc::op::lambda(Type resultType, Type inType1, Type inType1,
// function_ref<void(MutableArrayRef<BlockArgument>)> bodyBuilder) {

Value mlir::edsc::op::embed(
    Type result, ValueRange exposedValues,
    function_ref<void(MutableArrayRef<BlockArgument>)> bodyBuilder) {
  return ValueBuilder<EmbedOp>(
      result, exposedValues,
      [&](OpBuilder &nestedBuilder, Location nestedLoc,
          MutableArrayRef<BlockArgument> args) {
        if (bodyBuilder) {
          ScopedContext nestedContext(nestedBuilder, nestedLoc);
          OpBuilder::InsertionGuard guard(nestedBuilder);
          bodyBuilder(args);
        }
      });
}

void mlir::edsc::op::rise_return(Value returnValue) {
  OperationBuilder<rise::ReturnOp>(ValueRange{returnValue});
  return;
}

void mlir::edsc::op::out(Value writeTo, Value result) {
  OperationBuilder<rise::OutOp>(writeTo, result);
  return;
}

// TODO: This is what affineLoopBuilder looks like. The function is directly
// passed in the arguments here. Way better than before.

// void mlir::edsc::op::lambdaBuilder(ValueRange lbs, ValueRange ubs, int64_t
// step,
//                                   function_ref<void(Value)> bodyBuilder) {
//  // Fetch the builder and location.
//  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
//  OpBuilder &builder = ScopedContext::getBuilderRef();
//  Location loc = ScopedContext::getLocation();
//
//  // Create the actual loop and call the body builder, if provided, after
//  // updating the scoped context.
//  builder.create<AffineForOp>(
//      loc, lbs, builder.getMultiDimIdentityMap(lbs.size()), ubs,
//      builder.getMultiDimIdentityMap(ubs.size()), step,
//      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv) {
//        if (bodyBuilder) {
//          ScopedContext nestedContext(nestedBuilder, nestedLoc);
//          OpBuilder::InsertionGuard guard(nestedBuilder);
//          bodyBuilder(iv);
//        }
//        nestedBuilder.create<AffineTerminatorOp>(nestedLoc);
//      });
//}

// Value mapSeq(StringRef lowerTo, Nat n, DataType s, DataType t,
// function_ref<Value(void)> createLambda, Value array) {
//
//  auto lambda = createLambda();
//  Value mapSeq = mlir::edsc::op::mapSeq(n,s,t,lowerTo);
//
//  return ValueBuilder<ApplyOp>(
//      ArrayType::get(context, n, s),
//      mapSeq,
//      ValueRange{lambda, array});
//}

} // namespace edsc
} // namespace mlir
