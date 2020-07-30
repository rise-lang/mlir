//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rise/EDSC/HighLevel.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace edsc {

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::op;
using namespace mlir::edsc::type;
using namespace mlir::edsc::abstraction;
using namespace mlir::edsc::intrinsics;

// void mlir::edsc::highlevel::makeRiseTest(bool forwardDeclare = false) {
//  // generate funcs and calls and everything
//}

// A:MxN * B:NxK = C:MxK
Value mlir::edsc::highlevel::matrix_multiplication(int M, int N, int K, Value A,
                                                   Value B) {
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

  // clang-format off
  return mapSeq(arowType, [&](Value arow) {
    return (mapSeq(bcolType,  [&](Value bcol) {
      return (reduceSeq(elementType, [&](Value tuple, Value acc){
        return (embed3(elementType, ValueRange{fst(elementType, elementType, tuple),
                                               snd(elementType, elementType, tuple), acc},
                       [&](Value fst, Value snd, Value acc){
                         return acc * (fst + snd);
                       }));
      },literal(elementType, "0.000000"), zip(arowType.getSize(), elementType, elementType, arow, bcol)));
    }, transpose(B)));
  }, A);
  // clang-format on
}

Value mlir::edsc::highlevel::stencil(int N, int windowSize, int step,
                                     Value input) {
  int newN = (N + step - windowSize) / step;
  int padl = ceil((N - newN) / 2.0);
  int padr = ceil((N - newN) / 2.0);

  Value padded = padClamp(natType(padl), natType(padr), input);
  Value windowed = slide(natType(windowSize), natType(1), padded);

  return mapSeq(
      "scf", scalarF32Type(),
      [&](Value window) {
        return (reduceSeq("scf", scalarF32Type(), sumLambda(scalarF32Type()),
                          literal(scalarF32Type(), "0.000000"), window));
      },
      windowed);
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

Value mlir::edsc::highlevel::conv2D(Value input, Value kernel, int padl,
                                    int padr, int padt, int padb) {
  ArrayType inputHeight = input.getType().dyn_cast<ArrayType>();
  ArrayType inputWidth = inputHeight.getElementType().dyn_cast<ArrayType>();
  ArrayType kernelHeight = kernel.getType().dyn_cast<ArrayType>();
  ArrayType kernelWidth = kernelHeight.getElementType().dyn_cast<ArrayType>();
  ScalarType elementType = scalarF32Type();

  Value adjustedInput = input;
  if (padl != 0 || padr != 0 || padt != 0 || padb != 0) {
    adjustedInput = pad2D(natType(padt), natType(padb), natType(padl),
                          natType(padr), input);
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
      slided);
}

Value mlir::edsc::highlevel::conv2DSeparated(Value input, Value kernelH,
                                             Value kernelV, int padl, int padr,
                                             int padt, int padb) {
  ScalarType elementType = scalarF32Type();

  Value adjustedInput = input;
  if (padl != 0 || padr != 0 || padt != 0 || padb != 0) {
    adjustedInput = pad2D(natType(padt), natType(padb), natType(padl),
                          natType(padr), input);
  }

  // vertical
  Value slizzled = slide(natType(3), natType(1), adjustedInput);
  Value vertical = mapSeq(
      arrayType(slizzled.getType()
                    .dyn_cast<ArrayType>()
                    .getElementType()
                    .dyn_cast<ArrayType>()
                    .getElementType()
                    .dyn_cast<ArrayType>()
                    .getSize(),
                elementType),
      [&](Value arr) {
        Value transposed = transpose(arr);
        Value mapped = mapSeq(
            elementType,
            [&](Value nbh) {
              Value zipped = zip(nbh, kernelV);
              return reduceSeq(
                  elementType,
                  [&](Value tuple, Value acc) {
                    return embed3(scalarF32Type(),
                                  {fst(tuple), snd(tuple), acc},
                                  [&](Value fst, Value snd, Value acc) {
                                    Value res = acc + fst * snd;
                                    return res;
                                  });
                  },
                  literal(scalarF32Type(), "0.000000"), zipped);
            },
            transposed);
        return mapped;
      },
      slizzled);
  // horizontal

  Value horizontal = mapSeq(
      arrayType(vertical.getType().dyn_cast<ArrayType>().getSize(),
                elementType),
      [&](Value arr) {
        Value slizzled = slide(natType(3), natType(1), arr);
        Value mapped = mapSeq(
            elementType,
            [&](Value nbh) {
              nbh.getType().dump();
              Value zipped = zip(nbh, kernelH);
              Value reduced = reduceSeq(
                  elementType,
                  [&](Value tuple, Value acc) {
                    return embed3(scalarF32Type(),
                                  {fst(tuple), snd(tuple), acc},
                                  [&](Value fst, Value snd, Value acc) {
                                    Value res = acc + fst * snd;
                                    return res;
                                  });
                  },
                  literal(scalarF32Type(), "0.000000"), zipped);
              return reduced;
            },
            slizzled);
        return mapped;
      },
      vertical);

  return horizontal;
}

Value mlir::edsc::highlevel::conv2DTF(Value input, Value kernel) {

  ArrayType inputT = input.getType().dyn_cast<ArrayType>();
  ArrayType nestedInputT = inputT.getElementType().dyn_cast<ArrayType>();
  ArrayType nestedNestedInputT =
      nestedInputT.getElementType().dyn_cast<ArrayType>();

  ArrayType kernelT = kernel.getType().dyn_cast<ArrayType>();
  ArrayType nestedkernelT = kernelT.getElementType().dyn_cast<ArrayType>();

  Value reshapedInput = join(map(
      array2DType(nestedInputT.getSize(), nestedNestedInputT.getSize(),
                  scalarF32Type()),
      [&](Value array) {
        return map(
            arrayType(nestedNestedInputT.getSize(), scalarF32Type()),
            [&](Value elem) { return join(elem); }, array);
      },
      input));

  Value reshapedKernel = map(
      arrayType(nestedkernelT.getSize(), scalarF32Type()),
      [&](Value array) { return join(join(array)); }, kernel);

  ArrayType inputHeight = reshapedInput.getType().dyn_cast<ArrayType>();
  ArrayType inputWidth = inputHeight.getElementType().dyn_cast<ArrayType>();
  ArrayType kernelHeight = reshapedKernel.getType().dyn_cast<ArrayType>();
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
                         kernelWidth.getSize(), natType(1), reshapedInput);

  Value conv = mapSeq(
      array2DType(slided.getType()
                      .dyn_cast<ArrayType>()
                      .getElementType()
                      .dyn_cast<ArrayType>()
                      .getSize(),
                  natType(1), elementType),
      [&](Value arr) {
        return split(natType(1),
                     mapSeq(
                         elementType,
                         [&](Value slidingWindow) {
                           Value zipped = zip2D(slidingWindow, reshapedKernel);
                           Value joined = join(zipped);

                           return reduceSeq(
                               elementType,
                               [&](Value tuple, Value acc) {
                                 tuple.getType().dump();
                                 acc.getType().dump();
                                 return embed3(
                                     scalarF32Type(),
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
  return transpose(split(natType(1), conv));
}

Value mlir::edsc::highlevel::stencil2D(int M, int N, int outerWindowSize,
                                       int outerStep, int innerWindowSize,
                                       int innerStep, Value input) {
  ArrayType nestedType = arrayType(N, scalarF32Type());
  ArrayType inputType = arrayType(M, nestedType);

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
                         natType(padInnerl), natType(padInnerr), input);
  Value slizzled =
      slide2D(natType(outerWindowSize), natType(outerStep),
              natType(innerWindowSize), natType(innerStep), A_padded);

  return mapSeq2D(
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
}

Value mlir::edsc::utils::getFilledMemRef(ArrayRef<int64_t> shape,
                                         float fillValue, Value memref) {
  auto f32Type = FloatType::getF32(ScopedContext::getContext());

  SmallVector<Value, 4> lbs;
  SmallVector<Value, 4> ub;
  SmallVector<Value, 4> steps;
  for (size_t i = 0; i < shape.size(); i++) {
    lbs.push_back(std_constant_index(0));
    ub.push_back(std_constant_index(shape[i]));
    steps.push_back(std_constant_index(1));
  }
  Value cstf = std_constant_float(llvm::APFloat(fillValue), f32Type);
  if (!memref)
    memref = std_alloc(MemRefType::get(shape, f32Type, {}, 0));
  StdIndexedValue val(memref);

  loopNestBuilder(lbs, ub, steps, [&](auto ivs) {
    SmallVector<Value, 4> ivs_vector;
    for (int i = 0; i < shape.size(); i++) {
      ivs_vector.push_back(ivs[i]);
    }
    val(ivs_vector) = cstf;
  });
  return memref;
}

Value mlir::edsc::utils::getFilledMemRef(ArrayRef<int64_t> shape,
                                         Value memref) {
  auto f32Type = FloatType::getF32(ScopedContext::getContext());

  SmallVector<Value, 4> lbs;
  SmallVector<Value, 4> ub;
  SmallVector<Value, 4> steps;
  for (size_t i = 0; i < shape.size(); i++) {
    lbs.push_back(std_constant_index(0));
    ub.push_back(std_constant_index(shape[i]));
    steps.push_back(std_constant_index(1));
  }
  Value cst0f = std_constant_float(llvm::APFloat(0.0f), f32Type);
  Value cst1f = std_constant_float(llvm::APFloat(1.0f), f32Type);
  StdIndexedValue initVal(std_alloc(MemRefType::get({}, f32Type, {}, 0)));
  initVal = cst0f;

  if (!memref)
    memref = std_alloc(MemRefType::get(shape, f32Type, {}, 0));
  StdIndexedValue val(memref);

  mlir::edsc::loopNestBuilder(lbs, ub, steps, [&](auto ivs) {
    SmallVector<Value, 4> ivs_vector;
    for (int i = 0; i < shape.size(); i++) {
      ivs_vector.push_back(ivs[i]);
    }
    val(ivs_vector) = initVal();
    initVal = initVal + cst1f;
  });
  return memref;
}

} // namespace edsc
} // namespace mlir
