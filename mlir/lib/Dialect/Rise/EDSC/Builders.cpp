//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rise/EDSC/Builders.h"
#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
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

//===----------------------------------------------------------------------===//
// Rise Types
//===----------------------------------------------------------------------===//

FunType mlir::edsc::type::funType(Type in, Type out) {
  return FunType::get(ScopedContext::getContext(), in, out);
}

Nat mlir::edsc::type::natType(int val) {
  return Nat::get(ScopedContext::getContext(), val);
}

ArrayType mlir::edsc::type::arrayType(Nat size, DataType elemType) {
  return ArrayType::get(ScopedContext::getContext(), size, elemType);
}

ArrayType mlir::edsc::type::arrayType(int size, DataType elemType) {
  return arrayType(natType(size), elemType);
}

ArrayType mlir::edsc::type::array2DType(Nat outerSize, Nat innerSize,
                                        DataType elemType) {
  return arrayType(outerSize, arrayType(innerSize, elemType));
}

ArrayType mlir::edsc::type::array2DType(int outerSize, int innerSize,
                                        DataType elemType) {
  return array2DType(natType(outerSize), natType(innerSize), elemType);
}

ArrayType mlir::edsc::type::array3DType(Nat outerSize, Nat midSize,
                                        Nat innerSize, DataType elemType) {
  return array2DType(outerSize, midSize, arrayType(innerSize, elemType));
}

ArrayType mlir::edsc::type::array3DType(int outerSize, int midSize,
                                        int innerSize, DataType elemType) {
  return array2DType(outerSize, midSize,
                     arrayType(natType(innerSize), elemType));
}

Tuple mlir::edsc::type::tupleType(DataType lhs, DataType rhs) {
  return Tuple::get(ScopedContext::getContext(), lhs, rhs);
}

ScalarType mlir::edsc::type::scalarType(Type wrappedType) {
  return ScalarType::get(ScopedContext::getContext(), wrappedType);
}

ScalarType mlir::edsc::type::scalarF32Type() {
  return scalarType(FloatType::getF32(ScopedContext::getContext()));
}

ScalarType mlir::edsc::type::scalarF64Type() {
  return scalarType(FloatType::getF64(ScopedContext::getContext()));
}

//===----------------------------------------------------------------------===//
// Rise Operations: Core Lambda Calculus
//===----------------------------------------------------------------------===//

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

Value mlir::edsc::op::apply(DataType resultType, Value fun, ValueRange args) {
  return ValueBuilder<ApplyOp>(resultType, fun, args);
}

//===----------------------------------------------------------------------===//
// Rise Operations: Interoperability
//===----------------------------------------------------------------------===//

Value mlir::edsc::op::in(Value in, Type type) {
  assert(in.getType().isa<MemRefType>());
  return ValueBuilder<InOp>(type, in);
}

void mlir::edsc::op::out(Value writeTo, Value result) {
  OperationBuilder<rise::OutOp>(writeTo, result);
  return;
}

Value mlir::edsc::op::embed(
    Type result, ValueRange exposedValues,
    function_ref<Value(MutableArrayRef<BlockArgument>)> bodyBuilder) {
  return ValueBuilder<EmbedOp>(result, exposedValues,
                               [&](OpBuilder &nestedBuilder, Location nestedLoc,
                                   MutableArrayRef<BlockArgument> args) {
                                 ScopedContext nestedContext(nestedBuilder,
                                                             nestedLoc);
                                 OpBuilder::InsertionGuard guard(nestedBuilder);
                                 return bodyBuilder(args);
                               });
}

void mlir::edsc::op::rise_return(Value returnValue) {
  OperationBuilder<rise::ReturnOp>(ValueRange{returnValue});
  return;
}

void mlir::edsc::op::lowering_unit(function_ref<void()> bodyBuilder) {
  OperationBuilder<LoweringUnitOp>(
      [&](OpBuilder &nestedBuilder, Location nestedLoc) {
        ScopedContext nestedContext(nestedBuilder, nestedLoc);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        bodyBuilder();
      });
}

//===----------------------------------------------------------------------===//
// Rise Operations: Patterns
//===----------------------------------------------------------------------===//

Value mlir::edsc::op::mapSeq(DataType t,
                             function_ref<Value(BlockArgument)> bodyBuilder,
                             Value array) {
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();
  return mapSeq("affine", arrayT.getElementType(), t, bodyBuilder, array);
}

Value mlir::edsc::op::mapSeq(StringRef lowerTo, DataType t,
                             function_ref<Value(BlockArgument)> bodyBuilder,
                             Value array) {
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();
  return mapSeq(lowerTo, arrayT.getElementType(), t, bodyBuilder, array);
}

Value mlir::edsc::op::map(DataType t,
                          function_ref<Value(BlockArgument)> bodyBuilder,
                          Value array) {
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();
  return map(arrayT.getSize(), arrayT.getElementType(), t,
             lambda1(funType(arrayT.getElementType(), t), bodyBuilder), array);
}

Value mlir::edsc::op::reduceSeq(
    DataType t, function_ref<Value(BlockArgument, BlockArgument)> bodyBuilder,
    Value initializer, Value array) {
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();

  return reduceSeq(
      "affine", arrayT.getSize(), arrayT.getElementType(), t,
      lambda2(funType(arrayT.getElementType(), funType(t, t)), bodyBuilder),
      initializer, array);
}

Value mlir::edsc::op::zip(Value lhs, Value rhs) {
  MLIRContext *context = ScopedContext::getContext();
  ArrayType lhsType = lhs.getType().dyn_cast<ArrayType>();
  ArrayType rhsType = rhs.getType().dyn_cast<ArrayType>();
  return zip(lhsType.getSize(), lhsType.getElementType(),
             rhsType.getElementType(), lhs, rhs);
}

Value mlir::edsc::op::fst(Value tuple) {
  Tuple tupleType = tuple.getType().dyn_cast<Tuple>();
  return fst(tupleType.getFirst(), tupleType.getSecond(), tuple);
}

Value mlir::edsc::op::snd(Value tuple) {
  Tuple tupleType = tuple.getType().dyn_cast<Tuple>();
  return snd(tupleType.getFirst(), tupleType.getSecond(), tuple);
}

Value mlir::edsc::op::split(Nat n, Value array) {
  ArrayType arrayType = array.getType().dyn_cast<ArrayType>();

  return split(n, natType(arrayType.getSize().getIntValue() / n.getIntValue()),
               arrayType.getElementType(), array);
}

Value mlir::edsc::op::join(Value inArray) {
  ArrayType inArrayType = inArray.getType().dyn_cast<ArrayType>();
  ArrayType nestedArrayType =
      inArrayType.getElementType().dyn_cast<ArrayType>();

  return join(nestedArrayType.getSize(), inArrayType.getSize(),
              nestedArrayType.getElementType(), inArray);
}

Value mlir::edsc::op::transpose(Value array) {
  MLIRContext *context = ScopedContext::getContext();
  ArrayType outerArrayType = array.getType().dyn_cast<ArrayType>();
  ArrayType innerArrayType =
      outerArrayType.getElementType().dyn_cast<ArrayType>();

  return transpose(outerArrayType.getSize(), innerArrayType.getSize(),
                   innerArrayType.getElementType(), array);
}

Value mlir::edsc::op::slide(Nat windowSize, Nat step, Value array) {
  MLIRContext *context = ScopedContext::getContext();
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();
  int n = (arrayT.getSize().getIntValue() + step.getIntValue() -
           windowSize.getIntValue()) /
          step.getIntValue();

  return slide(natType(n), windowSize, step, arrayT.getElementType(), array);
}

Value mlir::edsc::op::padClamp(Nat l, Nat r, Value array) {
  MLIRContext *context = ScopedContext::getContext();
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();
  return padClamp(arrayT.getSize(), l, r, arrayT.getElementType(), array);
}

Value mlir::edsc::op::literal(DataType t, StringRef literal) {
  MLIRContext *context = ScopedContext::getContext();
  return ValueBuilder<LiteralOp>(t,
                                 LiteralAttr::get(context, t, literal.str()));
}

//===----------------------------------------------------------------------===//
// Rise high-level abstractions
//===----------------------------------------------------------------------===//

Value mlir::edsc::abstraction::mapSeq2D(
    DataType resultElemType, function_ref<Value(BlockArgument)> bodyBuilder,
    Value array2D) {
  return mapSeq2D("scf", resultElemType, bodyBuilder, array2D);
}

Value mlir::edsc::abstraction::mapSeq2D(
    StringRef lowerTo, DataType resultElemType,
    function_ref<Value(BlockArgument)> bodyBuilder, Value array2D) {
  ArrayType arrayT = array2D.getType().dyn_cast<ArrayType>();
  ArrayType nestedArrayT = arrayT.getElementType().dyn_cast<ArrayType>();
  return mapSeq(
      lowerTo, arrayType(nestedArrayT.getSize(), resultElemType),
      [&](Value array) {
        return mapSeq(lowerTo, resultElemType, bodyBuilder, array);
      },
      array2D);
}

Value mlir::edsc::abstraction::slide2D(Nat szOuter, Nat spOuter, Nat szInner,
                                       Nat spInner, Value array2DVal) {
  ArrayType arrayT = array2DVal.getType().dyn_cast<ArrayType>();
  ArrayType innerArrayType = arrayT.getElementType().dyn_cast<ArrayType>();
  int newInnerArraySize = (innerArrayType.getSize().getIntValue() +
                           spInner.getIntValue() - szInner.getIntValue()) /
                          spInner.getIntValue();

  Value afterFirstSlide = map(
      arrayType(newInnerArraySize,
                arrayType(szInner, innerArrayType.getElementType())),
      [&](Value innerArray) { return slide(szInner, spInner, innerArray); },
      array2DVal);

  Value afterSecondSlide = slide(szOuter, spOuter, afterFirstSlide);

  ArrayType afterSecondSlideType =
      afterSecondSlide.getType().dyn_cast<ArrayType>();
  ArrayType afterSecondSlideElementType =
      afterSecondSlideType.getElementType().dyn_cast<ArrayType>();
  ArrayType afterSecondSlideElementElementType =
      afterSecondSlideElementType.getElementType().dyn_cast<ArrayType>();

  Value transposed = map(
      arrayType(afterSecondSlideElementElementType.getSize(),
                arrayType(afterSecondSlideElementType.getSize(),
                          afterSecondSlideElementElementType.getElementType())),
      [&](auto args) { return transpose(args); }, afterSecondSlide);

  return transposed;
}

Value mlir::edsc::abstraction::pad2D(Nat lOuter, Nat rOuter, Nat lInner,
                                     Nat rInner, Value array) {
  ArrayType outerArrayType = array.getType().dyn_cast<ArrayType>();
  ArrayType innerArrayType =
      outerArrayType.getElementType().dyn_cast<ArrayType>();

  Value map_pad = map(
      arrayType(innerArrayType.getSize().getIntValue() + lInner.getIntValue() +
                    rInner.getIntValue(),
                innerArrayType.getElementType()),
      [&](Value innerArray) { return padClamp(lInner, rInner, innerArray); },
      array);
  return padClamp(lOuter, rOuter, map_pad);
}

Value mlir::edsc::abstraction::zip2D(Value array2DA, Value array2DB) {
  ArrayType arrayAType = array2DA.getType().dyn_cast<ArrayType>();
  ArrayType nestedAType = arrayAType.getElementType().dyn_cast<ArrayType>();
  ArrayType arrayBType = array2DB.getType().dyn_cast<ArrayType>();
  ArrayType nestedBType = arrayBType.getElementType().dyn_cast<ArrayType>();

  return map(
      arrayType(nestedAType.getSize(), tupleType(nestedAType.getElementType(),
                                                 nestedBType.getElementType())),
      [&](Value tuple) { return zip(fst(tuple), snd(tuple)); },
      zip(array2DA, array2DB));
}

using namespace mlir::edsc::op;
Value mlir::edsc::abstraction::sumLambda(ScalarType summandType) {
  return lambda(funType(summandType, funType(summandType, summandType)),
                [&](auto args) {
                  return (embed(summandType, ValueRange{args[0], args[1]},
                                [&](auto args) { return args[0] + args[1]; }));
                });
}

Value mlir::edsc::abstraction::multAndSumUpLambda(ScalarType summandType) {
  return lambda(funType(tupleType(summandType, summandType),
                        funType(summandType, summandType)),
                [&](auto args) {
                  return (
                      embed3(summandType,
                             ValueRange{fst(args[0]), snd(args[0]), args[1]},
                             [&](Value fst, Value snd, Value acc) {
                               return acc + (fst * snd);
                             }));
                });
}

//===----------------------------------------------------------------------===//
// Rise other convenience EDSC
//===----------------------------------------------------------------------===//

// one arg
Value mlir::edsc::op::lambda1(FunType lambdaType,
                              function_ref<Value(BlockArgument)> bodyBuilder) {
  return ValueBuilder<LambdaOp>(
      lambdaType, [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      MutableArrayRef<BlockArgument> args) {
        ScopedContext nestedContext(nestedBuilder, nestedLoc);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        return bodyBuilder(args[0]);
      });
}

// two args
Value mlir::edsc::op::lambda2(
    FunType lambdaType,
    function_ref<Value(BlockArgument, BlockArgument)> bodyBuilder) {
  return ValueBuilder<LambdaOp>(
      lambdaType, [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      MutableArrayRef<BlockArgument> args) {
        ScopedContext nestedContext(nestedBuilder, nestedLoc);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        return bodyBuilder(args[0], args[1]);
      });
}

// three args
Value mlir::edsc::op::lambda3(
    FunType lambdaType,
    function_ref<Value(BlockArgument, BlockArgument, BlockArgument)>
        bodyBuilder) {
  return ValueBuilder<LambdaOp>(
      lambdaType, [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      MutableArrayRef<BlockArgument> args) {
        ScopedContext nestedContext(nestedBuilder, nestedLoc);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        return bodyBuilder(args[0], args[1], args[2]);
      });
}

// four args
Value mlir::edsc::op::lambda4(FunType lambdaType,
                              function_ref<Value(BlockArgument, BlockArgument,
                                                 BlockArgument, BlockArgument)>
                                  bodyBuilder) {
  return ValueBuilder<LambdaOp>(
      lambdaType, [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      MutableArrayRef<BlockArgument> args) {
        ScopedContext nestedContext(nestedBuilder, nestedLoc);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        return bodyBuilder(args[0], args[1], args[2], args[3]);
      });
}

// five args
Value mlir::edsc::op::lambda5(
    FunType lambdaType,
    function_ref<Value(BlockArgument, BlockArgument, BlockArgument,
                       BlockArgument, BlockArgument)>
        bodyBuilder) {
  return ValueBuilder<LambdaOp>(
      lambdaType, [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      MutableArrayRef<BlockArgument> args) {
        ScopedContext nestedContext(nestedBuilder, nestedLoc);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        return bodyBuilder(args[0], args[1], args[2], args[3], args[4]);
      });
}

// six args
Value mlir::edsc::op::lambda6(
    FunType lambdaType,
    function_ref<Value(BlockArgument, BlockArgument, BlockArgument,
                       BlockArgument, BlockArgument, BlockArgument)>
        bodyBuilder) {
  return ValueBuilder<LambdaOp>(
      lambdaType, [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      MutableArrayRef<BlockArgument> args) {
        ScopedContext nestedContext(nestedBuilder, nestedLoc);
        OpBuilder::InsertionGuard guard(nestedBuilder);
        return bodyBuilder(args[0], args[1], args[2], args[3], args[4],
                           args[5]);
      });
}

// one arg
Value mlir::edsc::op::embed1(Type result, ValueRange exposedValues,
                             function_ref<Value(BlockArgument)> bodyBuilder) {
  return embed(result, exposedValues, [&](MutableArrayRef<BlockArgument> args) {
    return bodyBuilder(args[0]);
  });
}

// two args
Value mlir::edsc::op::embed2(
    Type result, ValueRange exposedValues,
    function_ref<Value(BlockArgument, BlockArgument)> bodyBuilder) {
  return embed(result, exposedValues, [&](MutableArrayRef<BlockArgument> args) {
    return bodyBuilder(args[0], args[1]);
  });
}

// three args
Value mlir::edsc::op::embed3(
    Type result, ValueRange exposedValues,
    function_ref<Value(BlockArgument, BlockArgument, BlockArgument)>
        bodyBuilder) {
  return embed(result, exposedValues, [&](MutableArrayRef<BlockArgument> args) {
    return bodyBuilder(args[0], args[1], args[2]);
  });
}

// four args
Value mlir::edsc::op::embed4(Type result, ValueRange exposedValues,
                             function_ref<Value(BlockArgument, BlockArgument,
                                                BlockArgument, BlockArgument)>
                                 bodyBuilder) {
  return embed(result, exposedValues, [&](MutableArrayRef<BlockArgument> args) {
    return bodyBuilder(args[0], args[1], args[2], args[3]);
  });
}

// five args
Value mlir::edsc::op::embed5(
    Type result, ValueRange exposedValues,
    function_ref<Value(BlockArgument, BlockArgument, BlockArgument,
                       BlockArgument, BlockArgument)>
        bodyBuilder) {
  return embed(result, exposedValues, [&](MutableArrayRef<BlockArgument> args) {
    return bodyBuilder(args[0], args[1], args[2], args[3], args[4]);
  });
}

// six args
Value mlir::edsc::op::embed6(
    Type result, ValueRange exposedValues,
    function_ref<Value(BlockArgument, BlockArgument, BlockArgument,
                       BlockArgument, BlockArgument, BlockArgument)>
        bodyBuilder) {
  return embed(result, exposedValues, [&](MutableArrayRef<BlockArgument> args) {
    return bodyBuilder(args[0], args[1], args[2], args[3], args[4], args[5]);
  });
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
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();
  return mapSeq(lowerTo, arrayT.getSize(), s, t, lambda, array);
}

Value mlir::edsc::op::mapSeq(StringRef lowerTo, DataType s, DataType t,
                             function_ref<Value(BlockArgument)> bodyBuilder,
                             Value array) {
  return mapSeq(lowerTo, s, t, lambda1(funType(s, t), bodyBuilder), array);
}

Value mlir::edsc::op::map(Nat n, DataType s, DataType t, Value lambda) {
  MLIRContext *context = ScopedContext::getContext();
  FunType mapType =
      FunType::get(context, FunType::get(context, s, t),
                   FunType::get(context, ArrayType::get(context, n, s),
                                ArrayType::get(context, n, t)));
  return ValueBuilder<MapOp>(mapType, NatAttr::get(context, n),
                             DataTypeAttr::get(context, s),
                             DataTypeAttr::get(context, t));
}

Value mlir::edsc::op::map(Nat n, DataType s, DataType t, Value lambda,
                          Value array) {
  MLIRContext *context = ScopedContext::getContext();
  Value mapOp = map(n, s, t, lambda);
  return ValueBuilder<ApplyOp>(ArrayType::get(context, n, t), mapOp,
                               ValueRange{lambda, array});
}

Value mlir::edsc::op::map(DataType t, Value lambda, Value array) {
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();
  return map(arrayT.getSize(), arrayT.getElementType(), t, lambda, array);
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

Value mlir::edsc::op::reduceSeq(
    StringRef lowerTo, DataType t,
    function_ref<Value(BlockArgument, BlockArgument)> bodyBuilder,
    Value initializer, Value array) {
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();

  return reduceSeq(
      lowerTo, arrayT.getSize(), arrayT.getElementType(), t,
      lambda2(funType(arrayT.getElementType(), funType(t, t)), bodyBuilder),
      initializer, array);
}

Value mlir::edsc::op::reduceSeq(StringRef lowerTo, DataType t, Value lambda,
                                Value initializer, Value array) {
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();

  return reduceSeq(lowerTo, arrayT.getSize(), arrayT.getElementType(), t,
                   lambda, initializer, array);
}

Value mlir::edsc::op::reduceSeq(DataType t, Value lambda, Value initializer,
                                Value array) {
  ArrayType arrayT = array.getType().dyn_cast<ArrayType>();

  return reduceSeq("scf", arrayT.getSize(), arrayT.getElementType(), t, lambda,
                   initializer, array);
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
  return apply(ArrayType::get(context, n, Tuple::get(context, s, t)), zipOp,
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

Value mlir::edsc::op::split(Nat n, Nat m, DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  return ValueBuilder<SplitOp>(
      funType(arrayType(natType(n.getIntValue() * m.getIntValue()), t),
              arrayType(m, arrayType(n, t))),
      NatAttr::get(context, n), NatAttr::get(context, m),
      DataTypeAttr::get(context, t));
}

Value mlir::edsc::op::split(Nat n, Nat m, DataType t, Value array) {
  Value splitOp = split(n, m, t);
  return ValueBuilder<ApplyOp>(arrayType(m, arrayType(n, t)), splitOp, array);
}

Value mlir::edsc::op::join(Nat n, Nat m, DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  return ValueBuilder<JoinOp>(
      funType(arrayType(m, arrayType(n, t)),
              arrayType(natType(n.getIntValue() * m.getIntValue()), t)),
      NatAttr::get(context, n), NatAttr::get(context, m),
      DataTypeAttr::get(context, t));
}

//// Do the join!
Value mlir::edsc::op::join(Nat n, Nat m, DataType t, Value inArray) {
  Value joinOp = join(n, m, t);
  return ValueBuilder<ApplyOp>(
      arrayType(natType(m.getIntValue() * n.getIntValue()), t), joinOp,
      inArray);
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

Value mlir::edsc::op::padClamp(Nat n, Nat l, Nat r, DataType t) {
  MLIRContext *context = ScopedContext::getContext();
  FunType padType = FunType::get(
      context, ArrayType::get(context, n, t),
      ArrayType::get(context,
                     Nat::get(context, l.getIntValue() + n.getIntValue() +
                                           r.getIntValue()),
                     t));
  return ValueBuilder<PadOp>(padType, NatAttr::get(context, n),
                             NatAttr::get(context, l), NatAttr::get(context, r),
                             DataTypeAttr::get(context, t));
}

Value mlir::edsc::op::padClamp(Nat n, Nat l, Nat r, DataType t, Value array) {
  MLIRContext *context = ScopedContext::getContext();

  Value pad = padClamp(n, l, r, t);
  return ValueBuilder<ApplyOp>(
      ArrayType::get(context,
                     Nat::get(context, l.getIntValue() + n.getIntValue() +
                                           r.getIntValue()),
                     t),
      pad, ValueRange{array});
}

} // namespace edsc
} // namespace mlir
