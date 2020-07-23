//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rise/EDSC/HighLevel.h"
#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Dialect/SCF/EDSC/Builders.h>

namespace mlir {
namespace edsc {

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::op;
using namespace mlir::edsc::type;
using namespace mlir::edsc::abstraction;
using namespace mlir::edsc::intrinsics;

// void mlir::edsc::highlevel::makeRiseProgram(ArrayRef<Value> inputs, Value
// output, function_ref<Value(BlockArgument)> bodyBuilder) {
//  return;
//}
//
// void mlir::edsc::highlevel::makeRiseTest(bool forwardDeclare = false) {
//  // generate funcs and calls and everything
//}

// A:MxN * B:NxK = C:MxK
void mlir::edsc::highlevel::matrix_multiplication(int M, int N, int K, Value A,
                                                  Value B, Value C) {
  // shape of A
  ArrayType AType = array2DType(M, N, scalarF32Type());
  rise::ArrayType arowType = AType.getElementType().dyn_cast<rise::ArrayType>();

  // shape of B
  rise::ArrayType BType = array2DType(N, K, scalarF32Type());
  rise::ArrayType BType_trans = array2DType(K, N, scalarF32Type());
  rise::ArrayType bcolType =
      BType_trans.getElementType().dyn_cast<rise::ArrayType>();

  rise::ScalarType elementType =
      arowType.getElementType().dyn_cast<rise::ScalarType>();

  // These have to be always on top!
  Value in_A = in(A, AType);
  Value in_B = in(B, BType);

  // clang-format off
  out(C, mapSeq(arowType, [&](Value arow) {
    return (mapSeq(bcolType,  [&](Value bcol) {
      return (reduceSeq(elementType, [&](Value tuple, Value acc){
        return (embed3(elementType, ValueRange{fst(elementType, elementType, tuple),
                                               snd(elementType, elementType, tuple), acc},
                       [&](Value fst, Value snd, Value acc){
                         return acc * (fst + snd);
                       }));
      },literal(elementType, "0.000000"), zip(arowType.getSize(), elementType, elementType, arow, bcol)));
    }, transpose(BType.getSize(), BType_trans.getSize(), elementType, in_B)));
  }, in_A));
  // clang-format on
  return;
}

void mlir::edsc::highlevel::stencil(int N, int windowSize, int step,
                                    Value input, Value output) {
  Value A = in(input, arrayType(N, scalarF32Type()));
  int newN = (N + step - windowSize) / step;
  int padl = ceil((N - newN) / 2.0);
  int padr = ceil((N - newN) / 2.0);

  Value padded = padClamp(natType(padl), natType(padr), A);
  Value windowed = slide(natType(windowSize), natType(1), padded);

  Value mapped = mapSeq(
      "scf", scalarF32Type(),
      [&](Value window) {
        return (reduceSeq("scf", scalarF32Type(), sumLambda(scalarF32Type()),
                          literal(scalarF32Type(), "0.000000"), window));
      },
      windowed);

  out(output, mapped);
}

Value mlir::edsc::highlevel::conv2D(Value input, Value kernel) {
  ArrayType inputHeight = input.getType().dyn_cast<ArrayType>();
  ArrayType inputWidth = inputHeight.getElementType().dyn_cast<ArrayType>();
  ArrayType kernelHeight = kernel.getType().dyn_cast<ArrayType>();
  ArrayType kernelWidth = kernelHeight.getElementType().dyn_cast<ArrayType>();

  int steplr = 1;
  int steptb = 1;

  int nWidth = (inputWidth.getSize().getIntValue() + steplr -
                kernelWidth.getSize().getIntValue()) /
               steplr;
  int nHeight = (inputHeight.getSize().getIntValue() + steptb -
                 kernelHeight.getSize().getIntValue()) /
                steptb;

  // If padding has to be different for l and r we pad 1 more on the left.
  int padInnerl = ceil((inputWidth.getSize().getIntValue() - nWidth) / 2.0);
  int padInnerr = floor((inputWidth.getSize().getIntValue() - nWidth) / 2.0);
  int padOuterl = ceil((inputHeight.getSize().getIntValue() - nHeight) / 2.0);
  int padOuterr = floor((inputHeight.getSize().getIntValue() - nHeight) / 2.0);

  return conv2D(input, kernel, padInnerl, padInnerr, padOuterl, padOuterr);
}

Value mlir::edsc::highlevel::conv2D(Value input, Value kernel, int padl, int padr, int padt, int padb) {
  ArrayType inputHeight = input.getType().dyn_cast<ArrayType>();
  ArrayType inputWidth = inputHeight.getElementType().dyn_cast<ArrayType>();
  ArrayType kernelHeight = kernel.getType().dyn_cast<ArrayType>();
  ArrayType kernelWidth = kernelHeight.getElementType().dyn_cast<ArrayType>();

  int steplr = 1;
  int steptb = 1;

  ScalarType elementType = scalarF32Type();

  Value adjustedInput = input;
  if (padl != 0 || padr != 0 || padt != 0 || padb != 0) {
    adjustedInput = pad2D(natType(padt), natType(padb),
                         natType(padl), natType(padr), input);
  }

  Value slided = slide2D(kernelHeight.getSize(), natType(1),
                         kernelWidth.getSize(), natType(1), adjustedInput);
  return mapSeq2D(
      elementType,
      [&](Value slidingWindow) {
        Value zipped = zip2D(slidingWindow, kernel);
        Value joined = join(zipped);

        return reduceSeq(
            elementType,
            [&](Value tuple, Value acc) {
              return embed3(scalarF32Type(), {fst(tuple), snd(tuple), acc},
                            [&](Value fst, Value snd, Value acc) {
                              Value res = acc + fst * snd;
//                              std_call("print_bin_op", ArrayRef<Type>(),
//                                       ValueRange{fst, snd, res});
                              return res;
                            });
            },
            literal(scalarF32Type(), "0.000000"), joined);
      },
      slided);
}


Value mlir::edsc::highlevel::conv2DTF(Value input, Value kernel) {
  ArrayType inputHeight = input.getType().dyn_cast<ArrayType>();
  ArrayType inputWidth = inputHeight.getElementType().dyn_cast<ArrayType>();
  ArrayType kernelHeight = kernel.getType().dyn_cast<ArrayType>();
  ArrayType kernelWidth = kernelHeight.getElementType().dyn_cast<ArrayType>();

  int steplr = 1;
  int steptb = 1;

  int nWidth = (inputWidth.getSize().getIntValue() + steplr -
                kernelWidth.getSize().getIntValue()) /
               steplr;
  int nHeight = (inputHeight.getSize().getIntValue() + steptb -
                 kernelHeight.getSize().getIntValue()) /
                steptb;

  // If padding has to be different for l and r we pad 1 more on the left.
  int padInnerl = ceil((inputWidth.getSize().getIntValue() - nWidth) / 2.0);
  int padInnerr = floor((inputWidth.getSize().getIntValue() - nWidth) / 2.0);
  int padOuterl = ceil((inputHeight.getSize().getIntValue() - nHeight) / 2.0);
  int padOuterr = floor((inputHeight.getSize().getIntValue() - nHeight) / 2.0);

  ScalarType elementType = scalarF32Type();
  //  Value padded = pad2D(natType(padOuterl), natType(padOuterr),
  //                       natType(padInnerl), natType(padInnerr), input);

  Value slided = slide2D(kernelHeight.getSize(), natType(1),
                         kernelWidth.getSize(), natType(1), input);
  slided.getType()
      .dyn_cast<ArrayType>()
      .getElementType()
      .dyn_cast<ArrayType>()
      .getSize();
  Value conv = mapSeq(
      array2DType(slided.getType()
                      .dyn_cast<ArrayType>()
                      .getElementType()
                      .dyn_cast<ArrayType>()
                      .getSize(),
                  natType(1), elementType),
      [&](Value arr) {
        return split(natType(1), mapSeq(
            elementType,
            [&](Value slidingWindow) {
              Value zipped = zip2D(slidingWindow, kernel);
              Value joined = join(zipped);

              // this split is a workaround
              return reduceSeq(
                  elementType,
                  [&](Value tuple, Value acc) {
                    return embed3(scalarF32Type(),
                                  {fst(tuple), snd(tuple), acc},
                                  [&](Value fst, Value snd, Value acc) {
                                    Value res = acc + fst * snd;
                                    //                              std_call("print_bin_op",
                                    //                              ArrayRef<Type>(),
                                    //                                       ValueRange{fst,
                                    //                                       snd,
                                    //                                       res});
                                    return res;
                                  });
                  },
                  literal(scalarF32Type(), "0.000000"), joined);
            },
            arr));
      },
      slided);
  ArrayType convType = conv.getType().dyn_cast<ArrayType>();
  convType.dump();
  ArrayType convNestedType = convType.getElementType().dyn_cast<ArrayType>();
  Value ypp = transpose(split(natType(1), conv));
  return ypp;
}


void mlir::edsc::highlevel::stencil2D(int M, int N, int outerWindowSize,
                                      int outerStep, int innerWindowSize,
                                      int innerStep, Value input,
                                      Value output) {
  ArrayType nestedType = arrayType(N, scalarF32Type());
  ArrayType inputType = arrayType(M, nestedType);
  Value A = in(input, inputType);

  int nInner =
      (inputType.getSize().getIntValue() + innerStep - innerWindowSize) /
      innerStep;
  int nOuter =
      (nestedType.getSize().getIntValue() + outerStep - outerWindowSize) /
      outerStep;

  // If padding has to be different for l and r we pad 1 more on the left.
  int padInnerl = ceil((M - nInner) / 2.0);
  int padInnerr = floor((M - nInner) / 2.0);
  int padOuterl = ceil((N - nOuter) / 2.0);
  int padOuterr = floor((N - nOuter) / 2.0);

  if (padInnerl != padInnerr)
    emitError(ScopedContext::getLocation())
        << "Due to an even sliding window in inner dimension we pad 1 value "
           "more "
           "on the lhs than on the rhs!";
  if (padOuterl != padOuterr)
    emitError(ScopedContext::getLocation())
        << "Due to an even sliding window in outer dimension we pad 1 value "
           "more "
           "on the left than on the right!";

  Value A_padded = pad2D(natType(padOuterl), natType(padOuterr),
                         natType(padInnerl), natType(padInnerr), A);
  Value slizzled =
      slide2D(natType(outerWindowSize), natType(outerStep),
              natType(innerWindowSize), natType(innerStep), A_padded);
  ArrayType slidedType = slizzled.getType().dyn_cast<ArrayType>();

  Value mapped = mapSeq2D(
      scalarF32Type(),
      [&](Value slidingWindow) {
        Value flattenedWindow = join(slidingWindow);
        return reduceSeq(
            "scf", scalarF32Type(),
            [&](Value arg1, Value arg2) {
              return embed2(scalarF32Type(), {arg1, arg2},
                            [&](Value arg1, Value arg2) {
                              Value res = arg1 + arg2;
                              //                              std_call("print_bin_op",
                              //                              ArrayRef<Type>(),
                              //                              ValueRange{arg1,
                              //                              arg2, res});
                              return res;
                            });
            },
            literal(scalarF32Type(), "0.000000"), flattenedWindow);
      },
      slizzled);

  out(output, mapped);
}

void mlir::edsc::highlevel::generateTest(int dims, ArrayRef<int64_t> inSizes,
                                         ArrayRef<int64_t> outSizes,
                                         FuncOp riseFun) {
  auto f32Type = FloatType::getF32(ScopedContext::getContext());

  if (!(dims == inSizes.size())) {
    emitError(ScopedContext::getLocation())
        << "Generating test failed. Dims has to match number of "
           "input and output sizes!";
    return;
  }

  auto inMemrefType = MemRefType::get(inSizes, f32Type, {}, 0);
  auto outMemrefType = MemRefType::get(outSizes, f32Type, {}, 0);

  SmallVector<Value, 4> lbs;
  SmallVector<Value, 4> ubs;
  SmallVector<Value, 4> steps;

  for (int i = 0; i < dims; i++) {
    lbs.push_back(std_constant_index(0));
    ubs.push_back(std_constant_index(inSizes[i]));
    steps.push_back(std_constant_index(1));
  }

  Value inMemref = std_alloc(inMemrefType);
  Value outMemref = std_alloc(outMemrefType);

  StdIndexedValue in(inMemref);
  StdIndexedValue out(outMemref);

  Value cst0f = std_constant_float(llvm::APFloat(0.0f), f32Type);
  Value cst1f = std_constant_float(llvm::APFloat(1.0f), f32Type);
  StdIndexedValue initVal(std_alloc(MemRefType::get({}, f32Type, {}, 0)));
  initVal = cst0f;

  // initializing the input with ascending values starting with 0
  Value ivs[dims];
  loopNestBuilder(lbs, ubs, steps, [&](auto ivs) {
    // bring ivs from ValueRange -> SmallVector to be usable by
    // TemplatedIndexedValue
    SmallVector<Value, 4> ivs_vector;
    for (int i = 0; i < dims; i++) {
      ivs_vector.push_back(ivs[i]);
    }
    in(ivs_vector) = initVal();
    initVal = initVal + cst1f;
  });

  if (riseFun) {
    std_call(riseFun, ValueRange{inMemref, outMemref});
    Value castedIn =
        std_memref_cast(inMemref, UnrankedMemRefType::get(f32Type, 0));
    std_call("print_memref_f32", ArrayRef<Type>(), ValueRange{castedIn});
    Value castedOut =
        std_memref_cast(outMemref, UnrankedMemRefType::get(f32Type, 0));
    std_call("print_memref_f32", ArrayRef<Type>(), ValueRange{castedOut});
  }
}

// TODO genTestND
void mlir::edsc::highlevel::generateTest(int dims, ArrayRef<int64_t> inSizesA,
                                         ArrayRef<int64_t> inSizesB,
                                         ArrayRef<int64_t> outSizes,
                                         FuncOp riseFun) {
  auto f32Type = FloatType::getF32(ScopedContext::getContext());

  if (!(dims == inSizesA.size())) {
    emitError(ScopedContext::getLocation())
        << "Generating test failed. Dims has to match number of "
           "input and output sizes!";
    return;
  }

  auto inAMemrefType = MemRefType::get(inSizesA, f32Type, {}, 0);
  auto inBMemrefType = MemRefType::get(inSizesB, f32Type, {}, 0);
  auto outMemrefType = MemRefType::get(outSizes, f32Type, {}, 0);

  SmallVector<Value, 4> lbs;
  SmallVector<Value, 4> ubA;
  SmallVector<Value, 4> ubB;
  SmallVector<Value, 4> steps;

  for (int i = 0; i < dims; i++) {
    lbs.push_back(std_constant_index(0));
    ubA.push_back(std_constant_index(inSizesA[i]));
    ubB.push_back(std_constant_index(inSizesB[i]));
    steps.push_back(std_constant_index(1));
  }

  Value inAMemref = std_alloc(inAMemrefType);
  Value inBMemref = std_alloc(inBMemrefType);
  Value outMemref = std_alloc(outMemrefType);

  StdIndexedValue inA(inAMemref);
  StdIndexedValue inB(inBMemref);
  StdIndexedValue out(outMemref);

  Value cst0f = std_constant_float(llvm::APFloat(0.0f), f32Type);
  Value cst1f = std_constant_float(llvm::APFloat(1.0f), f32Type);
  Value cst2f = std_constant_float(llvm::APFloat(2.0f), f32Type);

  // Init input A
  StdIndexedValue initAVal(std_alloc(MemRefType::get({}, f32Type, {}, 0)));
  initAVal = cst1f;

  // initializing the input with ascending values starting with 0
  loopNestBuilder(lbs, ubA, steps, [&](auto ivs) {
    // bring ivs from ValueRange -> SmallVector to be usable by
    // TemplatedIndexedValue
    SmallVector<Value, 4> ivs_vector;
    for (int i = 0; i < dims; i++) {
      ivs_vector.push_back(ivs[i]);
    }
    inA(ivs_vector) = initAVal();
    initAVal = initAVal + cst1f;
  });

  // init input B with only 1s
  StdIndexedValue initBVal(std_alloc(MemRefType::get({}, f32Type, {}, 0)));
  initBVal = cst1f;

  // initializing the input with 1s
  loopNestBuilder(lbs, ubB, steps, [&](auto ivs) {
    SmallVector<Value, 4> ivs_vector;
    for (int i = 0; i < dims; i++) {
      ivs_vector.push_back(ivs[i]);
    }
    inB(ivs_vector) = initBVal();
    //    initBVal = initBVal + cst1f;
  });

  if (riseFun) {
    std_call(riseFun, ValueRange{inAMemref, inBMemref, outMemref});
    Value castedInA =
        std_memref_cast(inAMemref, UnrankedMemRefType::get(f32Type, 0));
    std_call("print_memref_f32", ArrayRef<Type>(), ValueRange{castedInA});
    Value castedInB =
        std_memref_cast(inBMemref, UnrankedMemRefType::get(f32Type, 0));
    std_call("print_memref_f32", ArrayRef<Type>(), ValueRange{castedInB});

    Value castedOut =
        std_memref_cast(outMemref, UnrankedMemRefType::get(f32Type, 0));
    std_call("print_memref_f32", ArrayRef<Type>(), ValueRange{castedOut});
  }
}

} // namespace edsc
} // namespace mlir
