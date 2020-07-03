//===- builder-api-test.cpp - Tests for Declarative Builder APIs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-format off
// RUN: rise_highlevel_test | mlir-opt -split-input-file -convert-rise-to-imperative -canonicalize | FileCheck %s --check-prefix=IMPERATIVE
// RUN: rise_highlevel_test | mlir-opt -split-input-file -convert-rise-to-imperative -canonicalize --convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e stencil2D_test -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=STENCIL_2D_TEST
// RUN: rise_highlevel_test | mlir-opt -split-input-file -convert-rise-to-imperative -canonicalize --convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e pad2D_test -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=PAD_2D_TEST
//,/home/martin/development/phd/projects/MLIR/performance_measuring/dylib/measure_libi_no_mkl.so
// clang-format on

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Rise/EDSC/Builders.h"
#include "mlir/Dialect/Rise/EDSC/HighLevel.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/SCF/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#include "../../APITest.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

static MLIRContext &globalContext() {
  static bool init_once = []() {
    registerDialect<AffineDialect>();
    registerDialect<linalg::LinalgDialect>();
    registerDialect<scf::SCFDialect>();
    registerDialect<StandardOpsDialect>();
    registerDialect<vector::VectorDialect>();
    registerDialect<RiseDialect>();
    return true;
  }();
  (void)init_once;
  static thread_local MLIRContext context;
  context.allowUnregisteredDialects();
  return context;
}

static FuncOp makeFunction(StringRef name, ArrayRef<Type> results = {},
                           ArrayRef<Type> args = {}) {
  auto &ctx = globalContext();
  auto function = FuncOp::create(UnknownLoc::get(&ctx), name,
                                 FunctionType::get(args, results, &ctx));
  function.addEntryBlock();
  return function;
}

static FuncOp declareFunction(StringRef name, ArrayRef<Type> results = {},
                              ArrayRef<Type> args = {}) {
  auto &ctx = globalContext();
  auto function = FuncOp::create(UnknownLoc::get(&ctx), name,
                                 FunctionType::get(args, results, &ctx));
  return function;
}

// declare print_memref func only once
TEST_FUNC(declare_functions) {
  auto printMemref = declareFunction(
      "print_memref_f32", {},
      {UnrankedMemRefType::get(FloatType::getF32(&globalContext()), 0)});
//  auto printVal =
//      declareFunction("print_f32", {}, {FloatType::getF32(&globalContext())});
//  auto printBinOp = declareFunction("print_bin_op", {},
//                                    {FloatType::getF32(&globalContext()),
//                                     FloatType::getF32(&globalContext()),
//                                     FloatType::getF32(&globalContext())});

  printMemref.print(llvm::outs());
//  printVal.print(llvm::outs());
//  printBinOp.print(llvm::outs());

  printMemref.erase();
//  printVal.erase();
//  printBinOp.erase();
}

TEST_FUNC(build_and_lower_matrix_multiplication) {
  // A:MxN * B:NxK = C:MxK
  int64_t M = 32;
  int64_t N = 16;
  int64_t K = 64;

  auto f32Type = FloatType::getF32(&globalContext());

  auto f = makeFunction("mm", {},
                        {MemRefType::get({M, N}, f32Type, {}, 0),
                         MemRefType::get({N, K}, f32Type, {}, 0),
                         MemRefType::get({M, K}, f32Type, {}, 0)});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  Value A = f.getArgument(0);
  Value B = f.getArgument(1);
  Value C = f.getArgument(2);

  mlir::edsc::highlevel::matrix_multiplication(M, N, K, A, B, C);

  std_ret();

  // clang-format off
  // IMPERATIVE:       module {
  // IMPERATIVE-LABEL:   func @mm(
  // IMPERATIVE-SAME:             %[[VAL_0:.*]]: memref<32x16xf32>, %[[VAL_1:.*]]: memref<16x64xf32>, %[[VAL_2:.*]]: memref<32x64xf32>) {
  // IMPERATIVE:           %[[VAL_3:.*]] = constant 32 : index
  // IMPERATIVE:           %[[VAL_4:.*]] = constant 64 : index
  // IMPERATIVE:           %[[VAL_5:.*]] = constant 0.000000e+00 : f32
  // IMPERATIVE:           %[[VAL_6:.*]] = constant 0 : index
  // IMPERATIVE:           %[[VAL_7:.*]] = constant 16 : index
  // IMPERATIVE:           %[[VAL_8:.*]] = constant 1 : index
  // IMPERATIVE:           scf.for %[[VAL_9:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_8]] {
  // IMPERATIVE:             scf.for %[[VAL_10:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_8]] {
  // IMPERATIVE:               store %[[VAL_5]], %[[VAL_2]]{{\[}}%[[VAL_9]], %[[VAL_10]]] : memref<32x64xf32>
  // IMPERATIVE:               scf.for %[[VAL_11:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] {
  // IMPERATIVE:                 %[[VAL_12:.*]] = load %[[VAL_0]]{{\[}}%[[VAL_9]], %[[VAL_11]]] : memref<32x16xf32>
  // IMPERATIVE:                 %[[VAL_13:.*]] = load %[[VAL_1]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<16x64xf32>
  // IMPERATIVE:                 %[[VAL_14:.*]] = load %[[VAL_2]]{{\[}}%[[VAL_9]], %[[VAL_10]]] : memref<32x64xf32>
  // IMPERATIVE:                 %[[VAL_15:.*]] = addf %[[VAL_12]], %[[VAL_13]] : f32
  // IMPERATIVE:                 %[[VAL_16:.*]] = mulf %[[VAL_14]], %[[VAL_15]] : f32
  // IMPERATIVE:                 store %[[VAL_16]], %[[VAL_2]]{{\[}}%[[VAL_9]], %[[VAL_10]]] : memref<32x64xf32>
  // IMPERATIVE:               }
  // IMPERATIVE:             }
  // IMPERATIVE:           }
  // IMPERATIVE:           return
  // IMPERATIVE:         }
  // clang-format on

  f.print(llvm::outs());
  f.erase();
}

// TODO: This one computes the wrong result
TEST_FUNC(build_lower_and_execute_2Dstencil) {
  // A:MxN * B:NxK = C:MxK
  int64_t x_size = 7;
  int64_t y_size = 5;

  auto f32Type = FloatType::getF32(&globalContext());

  auto f = makeFunction("stencil2D", {},
                        {MemRefType::get({x_size, y_size}, f32Type, {}, 0),
                         MemRefType::get({x_size, y_size}, f32Type, {}, 0)});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  Value input = f.getArgument(0);
  Value output = f.getArgument(1);

  mlir::edsc::highlevel::stencil2D(x_size, y_size, 5, 1, 3, 1, input, output);
  std_ret();

  // generate test
  auto testFun = makeFunction("stencil2D_test", {}, {});
  OpBuilder test_builder(testFun.getBody());
  ScopedContext test_scope(test_builder, testFun.getLoc());
  mlir::edsc::highlevel::generateTest(2, {x_size, y_size}, {x_size, y_size}, f);
  std_ret();
// clang-format off

// STENCIL_2D_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [7, 5] strides = [5, 1] data =
// STENCIL_2D_TEST:       {{\[\[}}50,   60,   75,   90,   100],
// STENCIL_2D_TEST:        [95,   105,   120,   135,   145],
// STENCIL_2D_TEST:        [155,   165,   180,   195,   205],
// STENCIL_2D_TEST:        [230,   240,   255,   270,   280],
// STENCIL_2D_TEST:        [305,   315,   330,   345,   355],
// STENCIL_2D_TEST:        [365,   375,   390,   405,   415],
// STENCIL_2D_TEST:        [410,   420,   435,   450,   460]]
// clang-format on

  f.print(llvm::outs());
  testFun.print(llvm::outs());

  f.erase();
  testFun.erase();
}

using namespace mlir::edsc::op;
using namespace mlir::edsc::type;
using namespace mlir::edsc::highlevel;
using namespace mlir::edsc::abstraction;

TEST_FUNC(test_slide2d) {
  int64_t M = 7;
  int64_t N = 5;
  int slideOuter = 5;
  int slideInner = 3;
  auto f32Type = FloatType::getF32(&globalContext());

  auto f = makeFunction("slide2D", {},
                        {MemRefType::get({M, N}, f32Type, {}, 0),
                         MemRefType::get({M-2, N-4, slideOuter, slideInner},
                                         f32Type, {}, 0)});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  Value input = f.getArgument(0);
  Value output = f.getArgument(1);

  Value inn = in(input, arrayType(M, arrayType(N, scalarF32Type())));
  Value slizzled = slide2D(natType(slideOuter), natType(1), natType(slideInner), natType(1), inn);

  Value mapped = mapSeq2D(
      array2DType(slideOuter, slideInner, scalarF32Type()),
      [&](Value arr2D) {
        return mapSeq2D(
            scalarF32Type(),
            [&](Value elem) {
              return embed1(scalarF32Type(), elem, [&](Value elem) {
                Value cst = std_constant_float(llvm::APFloat(0.0f), f32Type);
                return elem + cst;
              });
            },
            arr2D);
      },
      slizzled);

  out(output, mapped);

  std_ret();

  // generate test
  auto testFun = makeFunction("slide2D_test", {}, {});
  OpBuilder test_builder(testFun.getBody());
  ScopedContext test_scope(test_builder, testFun.getLoc());
  mlir::edsc::highlevel::generateTest(
      2, {M, N}, {M - 2, N - 4, slideOuter, slideInner}, f);
  std_ret();

  testFun.print(llvm::outs());
  f.print(llvm::outs());

  testFun.erase();
  f.erase();
}

TEST_FUNC(test_pad2d) {
  int64_t width = 5;
  int64_t height = 7;
  int padInnerl = 1;
  int padInnerr = 1;
  int padOuterl = 2;
  int padOuterr = 2;
  int64_t outWidth = padInnerl + width + padInnerr;
  int64_t outHeight = padOuterl + height + padOuterr;

  auto f32Type = FloatType::getF32(&globalContext());

  auto f =
      makeFunction("pad2D", {},
                   {MemRefType::get({height, width}, f32Type, {}, 0),
                    MemRefType::get({outHeight, outWidth}, f32Type, {}, 0)});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  Value input = f.getArgument(0);
  Value output = f.getArgument(1);

  Value inn = in(input, arrayType(height, arrayType(width, scalarF32Type())));
  Value padded = pad2D(natType(padOuterl), natType(padOuterr),
                       natType(padInnerl), natType(padInnerr), inn);

  ArrayType paddedType = padded.getType().dyn_cast<ArrayType>();
  ArrayType innerType = paddedType.getElementType().dyn_cast<ArrayType>();

  Value mapped = mapSeq2D(
      scalarF32Type(),
      [&](Value elem) {
        return embed1(scalarF32Type(), elem, [&](Value elem) {
          Value cst = std_constant_float(llvm::APFloat(0.0f), f32Type);
          return elem + cst;
        });
      },
      padded);

  out(output, mapped);

  std_ret();

  // generate test
  auto testFun = makeFunction("pad2D_test", {}, {});
  OpBuilder test_builder(testFun.getBody());
  ScopedContext test_scope(test_builder, testFun.getLoc());
  mlir::edsc::highlevel::generateTest(2, {height, width}, {outHeight, outWidth},
                                      f);
  std_ret();

// PAD_2D_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [7, 5] strides = [5, 1] data =
// PAD_2D_TEST:       {{\[\[}}0,   1,   2,   3,   4],
// PAD_2D_TEST:        [5,   6,   7,   8,   9],
// PAD_2D_TEST:        [10,   11,   12,   13,   14],
// PAD_2D_TEST:        [15,   16,   17,   18,   19],
// PAD_2D_TEST:        [20,   21,   22,   23,   24],
// PAD_2D_TEST:        [25,   26,   27,   28,   29],
// PAD_2D_TEST:        [30,   31,   32,   33,   34]]
// PAD_2D_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [11, 7] strides = [7, 1] data =
// PAD_2D_TEST:       {{\[\[}}0,   0,   1,   2,   3,   4,   4],
// PAD_2D_TEST:        [0,   0,   1,   2,   3,   4,   4],
// PAD_2D_TEST:        [0,   0,   1,   2,   3,   4,   4],
// PAD_2D_TEST:        [5,   5,   6,   7,   8,   9,   9],
// PAD_2D_TEST:        [10,   10,   11,   12,   13,   14,   14],
// PAD_2D_TEST:        [15,   15,   16,   17,   18,   19,   19],
// PAD_2D_TEST:        [20,   20,   21,   22,   23,   24,   24],
// PAD_2D_TEST:        [25,   25,   26,   27,   28,   29,   29],
// PAD_2D_TEST:        [30,   30,   31,   32,   33,   34,   34],
// PAD_2D_TEST:        [30,   30,   31,   32,   33,   34,   34],
// PAD_2D_TEST:        [30,   30,   31,   32,   33,   34,   34]]


  f.print(llvm::outs());
  testFun.print(llvm::outs());

  f.erase();
  testFun.erase();
}

int main() {
  RUN_TESTS();
  return 0;
}
