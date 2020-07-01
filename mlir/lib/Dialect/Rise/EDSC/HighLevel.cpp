//===- Builders.cpp - MLIR Declarative Builder Classes --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rise/EDSC/HighLevel.h"
#include "mlir/Dialect/Rise/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include <mlir/Dialect/SCF/EDSC/Builders.h>

namespace mlir {
namespace edsc {

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::op;
using namespace mlir::edsc::type;
using namespace mlir::edsc::abstraction;
using namespace mlir::edsc::intrinsics;

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
                         return(std_mulf(
                             acc, std_addf(fst, snd)));
                       }));
      },literal(elementType, "0.000000"), zip(arowType.getSize(), elementType, elementType, arow, bcol)));
    }, transpose(BType.getSize(), BType_trans.getSize(), elementType, in_B)));
  }, in_A));
  // clang-format on
  return;
}

void mlir::edsc::highlevel::stencil(int n, Value input, Value output) {
  int slidingWindow = 3;

  Value A = in(input, arrayType(n, scalarF32Type()));

  Value padded = padClamp(natType(1), natType(1), A);
  Value windowed = slide(natType(slidingWindow), natType(1), padded);

  Value mapped = mapSeq(
      "loop", arrayType(slidingWindow, scalarF32Type()), scalarF32Type(),
      [&](Value window) {
        return (reduceSeq("loop", scalarF32Type(), sumLambda(scalarF32Type()),
                          literal(scalarF32Type(), "0.000000"), window));
      },
      windowed);

  out(output, mapped);
}

void mlir::edsc::highlevel::stencil2D(int n, Value input, Value output) {
  int slidingWindow = 3;

  Value A = in(input, arrayType(n, arrayType(n, scalarF32Type())));

  Value slizzled = slide2d(natType(3), natType(1), natType(5), natType(1), A);
  ArrayType slidedType =
      arrayType(n - 2, arrayType(natType(n - 4),
                                 arrayType(3, arrayType(5, scalarF32Type()))));

  Value mapped = mapSeq(
      "loop", slidedType.getElementType(), arrayType(n - 4, scalarF32Type()),
      [&](auto nestedArray) {
        return mapSeq(
            "loop", arrayType(3, arrayType(5, scalarF32Type())),
            scalarF32Type(),
            [&](auto slidingWindow) {
              Value flattenedWindow = join(slidingWindow);
              return reduceSeq(
                  "loop", scalarF32Type(), sumLambda(scalarF32Type()),
                  literal(scalarF32Type(), "0.000000"), flattenedWindow);
            },
            nestedArray);
      },
      slizzled);

  //  Value padded = padClamp(natType(1), natType(1), A);
  //  Value windowed = slide(natType(slidingWindow), natType(1), padded);

  //  Value mapped = mapSeq("loop", arrayType(slidingWindow, scalarF32Type()),
  //  scalarF32Type(),  [&](auto args) {
  //    return (reduceSeq("loop", sumLambda(scalarF32Type()),
  //    literal(scalarF32Type(), "0.000000"), args[0]));
  //  }, windowed);

  //  Value mapped = mapSeq("loop", arrayType(slidingWindow, scalarF32Type()),
  //  scalarF32Type(), lambda(funtype(arrayType(slidingWindow, scalarF32Type()),
  //  scalarF32Type()), [&](auto args) {
  //    return (reduceSeq("loop", sumLambda(scalarF32Type()),
  //    literal(scalarF32Type(), "0.000000"), args[0]));
  //  }), windowed);

  out(output, mapped);
}

void mlir::edsc::highlevel::generateTest(int dims, ArrayRef<int64_t> inSizes, ArrayRef<int64_t> outSizes, FuncOp riseFun) {
  auto f32Type = FloatType::getF32(ScopedContext::getContext());

  if (!((dims == inSizes.size()) && (dims == outSizes.size()))) {
    emitError(ScopedContext::getLocation()) << "Generating test failed. Dims has to match number of "
                      "input and output sizes!";
    return;
  }

  auto inMemrefType = MemRefType::get(inSizes, f32Type, {}, 0);
  auto outMemrefType = MemRefType::get(outSizes, f32Type, {}, 0);

  //  Value[dims] lbs = std_constant_index(inSizes);
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

  TemplatedIndexedValue<std_load, std_store> in(inMemref);
  TemplatedIndexedValue<std_load, std_store> out(outMemref);

  Value cst0f = std_constant_float(llvm::APFloat(0.0f), f32Type);
  Value cst1f = std_constant_float(llvm::APFloat(1.0f), f32Type);
  TemplatedIndexedValue<std_load, std_store> initVal(
      std_alloc(MemRefType::get({}, f32Type, {}, 0)));
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
    Value castedOut =
        std_memref_cast(outMemref, UnrankedMemRefType::get(f32Type, 0));
    std_call("print_memref_f32", ArrayRef<Type>(), ValueRange{castedOut});
  }
}

} // namespace edsc
} // namespace mlir
