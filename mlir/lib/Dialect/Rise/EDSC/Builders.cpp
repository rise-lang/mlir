//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rise/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include <iostream>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir::rise;
using namespace mlir::edsc::type;
using namespace mlir::edsc::op;

namespace mlir {
namespace edsc {

// Types
FunType mlir::edsc::type::funtype(Type in, Type out) {
  return FunType::get(ScopedContext::getContext(), in, out);
}

Nat mlir::edsc::type::nat(int val) {
  return Nat::get(ScopedContext::getContext(), val);
}

ArrayType mlir::edsc::type::array(Nat size, DataType elemType) {
  return ArrayType::get(ScopedContext::getContext(), size, elemType);
}

ArrayType mlir::edsc::type::array(int size, DataType elemType) {
  return array(nat(size), elemType);
}

ArrayType mlir::edsc::type::array2D(Nat outerSize, Nat innerSize,
                                    DataType elemType) {
  return array(outerSize, array(innerSize, elemType));
}

ArrayType mlir::edsc::type::array2D(int outerSize, int innerSize,
                                    DataType elemType) {
  return array2D(nat(outerSize), nat(innerSize), elemType);
}

ArrayType mlir::edsc::type::array3D(Nat outerSize, Nat midSize, Nat innerSize,
                                    DataType elemType) {
  return array2D(outerSize, midSize, array(innerSize, elemType));
}

ArrayType mlir::edsc::type::array3D(int outerSize, int midSize, int innerSize,
                                    DataType elemType) {
  return array2D(outerSize, midSize, array(nat(innerSize), elemType));
}

Tuple mlir::edsc::type::tuple(DataType lhs, DataType rhs) {
  return Tuple::get(ScopedContext::getContext(), lhs, rhs);
}

ScalarType mlir::edsc::type::scalar(Type wrappedType) {
  return ScalarType::get(ScopedContext::getContext(), wrappedType);
}

ScalarType mlir::edsc::type::scalarF32() {
  return scalar(FloatType::getF32(ScopedContext::getContext()));
}

ScalarType mlir::edsc::type::scalarF64() {
  return scalar(FloatType::getF64(ScopedContext::getContext()));
}

// Operations
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

Value mlir::edsc::op::reduceSeq(StringRef lowerTo, Value lambda,
                                Value initializer, Value array) {
  ArrayType arrayType = array.getType().dyn_cast<ArrayType>();
  return reduceSeq(lowerTo, arrayType.getSize(), arrayType.getElementType(),
                   arrayType.getElementType(), lambda, initializer, array);
}

Value mlir::edsc::op::reduceSeq(
    StringRef lowerTo, DataType s, DataType t,
    function_ref<Value(MutableArrayRef<BlockArgument>)> bodyBuilder,
    Value initializer, Value array) {
  ArrayType arrayType = array.getType().dyn_cast<ArrayType>();

  return reduceSeq(lowerTo, arrayType.getSize(), arrayType.getElementType(),
                   arrayType.getElementType(),
                   lambda(funtype(s, funtype(s, t)), bodyBuilder), initializer,
                   array);
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

Value mlir::edsc::op::zip(Value lhs, Value rhs) {
  MLIRContext *context = ScopedContext::getContext();
  ArrayType lhsType = lhs.getType().dyn_cast<ArrayType>();
  ArrayType rhsType = rhs.getType().dyn_cast<ArrayType>();
  return zip(lhsType.getSize(), lhsType.getElementType(),
             rhsType.getElementType(), lhs, rhs);
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

Value mlir::edsc::op::fst(Value tuple) {
  Tuple tupleType = tuple.getType().dyn_cast<Tuple>();
  return fst(tupleType.getFirst(), tupleType.getSecond(), tuple);
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

Value mlir::edsc::op::snd(Value tuple) {
  Tuple tupleType = tuple.getType().dyn_cast<Tuple>();
  return snd(tupleType.getFirst(), tupleType.getSecond(), tuple);
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

Value mlir::edsc::op::transpose(Value array) {
  MLIRContext *context = ScopedContext::getContext();
  ArrayType outerArrayType = array.getType().dyn_cast<ArrayType>();
  ArrayType innerArrayType =
      outerArrayType.getElementType().dyn_cast<ArrayType>();

  return transpose(innerArrayType.getSize(), outerArrayType.getSize(),
                   innerArrayType.getElementType(), array);
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

Value mlir::edsc::op::slide(Nat sz, Nat sp, Value array) {
  MLIRContext *context = ScopedContext::getContext();
  ArrayType arrayType = array.getType().dyn_cast<ArrayType>();
  int n = (arrayType.getSize().getIntValue() + sp.getIntValue() -
           sz.getIntValue()) /
          sp.getIntValue();

  return slide(nat(n), sz, sp, arrayType.getElementType(), array);
}

Value mlir::edsc::op::padClamp(Nat n, Nat l, Nat r, DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  FunType padType = FunType::get(
      context, t,
      FunType::get(
          context, ArrayType::get(context, n, t),
          ArrayType::get(context,
                         Nat::get(context, l.getIntValue() + n.getIntValue() +
                                               r.getIntValue()),
                         t)));
  return ValueBuilder<PadOp>(padType, NatAttr::get(context, n),
                             NatAttr::get(context, l), NatAttr::get(context, r),
                             DataTypeAttr::get(context, t));
}

Value mlir::edsc::op::padClamp(Nat n, Nat l, Nat r, DataType t, Value array) {
  MLIRContext *context = ScopedContext::getContext();
  Value cst0 = mlir::edsc::intrinsics::std_constant_float(
      llvm::APFloat(7.0f), FloatType::getF32(context));

  Value pad = padClamp(n, l, r, t);
  return ValueBuilder<ApplyOp>(
      ArrayType::get(context,
                     Nat::get(context, l.getIntValue() + n.getIntValue() +
                                           r.getIntValue()),
                     t),
      pad, ValueRange{cst0, array});
}

Value mlir::edsc::op::padClamp(Nat l, Nat r, Value array) {
  MLIRContext *context = ScopedContext::getContext();
  ArrayType arrayType = array.getType().dyn_cast<ArrayType>();
  return padClamp(arrayType.getSize(), l, r, arrayType.getElementType(), array);
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
  return ValueBuilder<ApplyOp>(ArrayType::get(context, n, t), mapSeq,
                               ValueRange{lambda, array});
}

Value mlir::edsc::op::mapSeq(StringRef lowerTo, DataType s, DataType t,
                             Value lambda, Value array) {
  ArrayType arrayType = array.getType().dyn_cast<ArrayType>();
  return mapSeq(lowerTo, arrayType.getSize(), s, t, lambda, array);
}

Value mlir::edsc::op::mapSeq(
    StringRef lowerTo, DataType s, DataType t,
    function_ref<Value(MutableArrayRef<BlockArgument>)> bodyBuilder,
    Value array) {
  return mapSeq(lowerTo, s, t, lambda(funtype(s, t), bodyBuilder), array);
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

// // I want to have an embed which does not return a Value in its body. However
// the template instantiation here is ambiguous for some reason.
// Value mlir::edsc::op::embed(
//    Type result, ValueRange exposedValues,
//    function_ref<void(MutableArrayRef<BlockArgument>)> bodyBuilder) {
//  return ValueBuilder<EmbedOp>(
//      result, exposedValues,
//      [&](OpBuilder &nestedBuilder, Location nestedLoc,
//          MutableArrayRef<BlockArgument> args) -> void {
//          ScopedContext nestedContext(nestedBuilder, nestedLoc);
//          OpBuilder::InsertionGuard guard(nestedBuilder);
//          bodyBuilder(args);
//          return;
//      });
//}

Value mlir::edsc::op::embed(
    Type result, ValueRange exposedValues,
    function_ref<Value(MutableArrayRef<BlockArgument>)> bodyBuilder) {
  return ValueBuilder<EmbedOp>(
      result, exposedValues,
      [&](OpBuilder &nestedBuilder, Location nestedLoc,
          MutableArrayRef<BlockArgument> args) -> Value {
        ScopedContext nestedContext(nestedBuilder, nestedLoc);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        return bodyBuilder(args);
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

// Abstractions

// How do I include everything here correctly to have + of Values overloaded?
Value mlir::edsc::abstraction::sumLambda(ScalarType summandType) {
  return lambda(
      funtype(summandType, funtype(summandType, summandType)), [&](auto args) {
        return (
            embed(summandType, ValueRange{args[0], args[1]}, [&](auto args) {
              return (mlir::edsc::intrinsics::std_addf(args[0], args[1]));
            }));
      });
}

Value mlir::edsc::abstraction::multAndSumUpLambda(ScalarType summandType) {
  return lambda(
      funtype(tuple(summandType, summandType),
              funtype(summandType, summandType)),
      [&](auto args) {
        return (embed(
            summandType,
            ValueRange{fst(summandType, summandType, args[0]),
                       snd(summandType, summandType, args[0]), args[1]},
            [&](auto args) {
              return (mlir::edsc::intrinsics::std_mulf(
                  args[0], mlir::edsc::intrinsics::std_addf(args[1], args[2])));
            }));
      });
}

// ArrayType::get(context, n, ArrayType::get(context, sz, t))
// Value mlir::edsc::abstraction::slide2d(Nat szOuter, Nat stOuter, Nat szInner,
// Nat stInner, Value array2DVal) {
//  ArrayType arrayType = array2DVal.getType().dyn_cast<ArrayType>();
//  ArrayType innerArrayType = arrayType.getElementType().dyn_cast<ArrayType>();
//
//
//  mapSeq("loop", lambda(funtype(arrayType.getElementType(), array2D())))
//  return
//}

Value mlir::edsc::abstraction::slide2d(Nat szOuter, Nat spOuter, Nat szInner,
                                       Nat spInner, Value array2DVal) {
  ArrayType arrayType = array2DVal.getType().dyn_cast<ArrayType>();
  ArrayType innerArrayType = arrayType.getElementType().dyn_cast<ArrayType>();
  int newInnerArraySize = (innerArrayType.getSize().getIntValue() +
                           spInner.getIntValue() - szInner.getIntValue()) /
                          spInner.getIntValue();
  std::cout << "innernewsizeL" << newInnerArraySize << std::flush;
  Value afterFirstSlide = mapSeq(
      "loop", innerArrayType,
      array(nat(newInnerArraySize),
            array(szInner, innerArrayType.getElementType())),
      [&](auto args) { return slide(szInner, spInner, args[0]); }, array2DVal);

  Value afterSecondSlide = slide(szOuter, spOuter, afterFirstSlide);

  ArrayType afterSecondSlideType =
      afterSecondSlide.getType().dyn_cast<ArrayType>();
  ArrayType afterSecondSlideElementType =
      afterSecondSlideType.getElementType().dyn_cast<ArrayType>();
  ArrayType afterSecondSlideElementElementType =
      afterSecondSlideElementType.getElementType().dyn_cast<ArrayType>();

  Value transposed = mapSeq("loop", afterSecondSlideElementType, array(afterSecondSlideElementElementType.getSize(), array(afterSecondSlideElementType.getSize(), afterSecondSlideElementElementType.getElementType())), [&](auto args){
    return transpose(args[0]);
  }, afterSecondSlide);

  return transposed;
}

// def slide2D(szOuter: Nat, stOuter: Nat, szInner: Nat, stInner: Nat): Expr =
// map(slide(szInner)(stInner)) >> slide(szOuter)(stOuter) >> map(transpose)

} // namespace edsc
} // namespace mlir
