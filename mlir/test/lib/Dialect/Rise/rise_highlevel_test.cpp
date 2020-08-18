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
// RUN: rise_highlevel_test | mlir-opt -split-input-file -convert-rise-to-imperative -canonicalize --convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e zip2D_test -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=ZIP_2D_TEST
// RUN: rise_highlevel_test | mlir-opt -split-input-file -convert-rise-to-imperative -canonicalize --convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e conv2D_test -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=CONV_2D_TEST
//
// Drop measurelib in folder where the other lib is.
//,/home/martin/development/phd/projects/MLIR/performance_measuring/dylib/measure_libi_no_mkl.so
// echo $PATH | FileCheck %s --check-prefix=IMPERATIVE
// clang-format on

#include <iostream>
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
using namespace mlir::edsc::op;
using namespace mlir::edsc::type;
using namespace mlir::edsc::highlevel;
using namespace mlir::edsc::abstraction;
using namespace mlir::edsc::utils;

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
  auto printVal =
      declareFunction("print_f32", {}, {FloatType::getF32(&globalContext())});
  auto printBinOp = declareFunction("print_bin_op", {},
                                    {FloatType::getF32(&globalContext()),
                                     FloatType::getF32(&globalContext()),
                                     FloatType::getF32(&globalContext())});

  printMemref.print(llvm::outs());
  printVal.print(llvm::outs());
  printBinOp.print(llvm::outs());

  printMemref.erase();
  printVal.erase();
  printBinOp.erase();
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

  makeRiseProgram(C, A, B)([&](Value A, Value B) {
    return mlir::edsc::highlevel::matrix_multiplication(M, N, K, A, B);
  });
  std_ret();

  // clang-format off
// IMPERATIVE-LABEL:   func @mm(
// IMPERATIVE-SAME:             %[[VAL_0:.*]]: memref<32x16xf32>,
// IMPERATIVE-SAME:             %[[VAL_1:.*]]: memref<16x64xf32>,
// IMPERATIVE-SAME:             %[[VAL_2:.*]]: memref<32x64xf32>) {
// IMPERATIVE:           %[[VAL_3:.*]] = constant 32 : index
// IMPERATIVE:           %[[VAL_4:.*]] = constant 64 : index
// IMPERATIVE:           %[[VAL_5:.*]] = constant 0.000000e+00 : f32
// IMPERATIVE:           %[[VAL_6:.*]] = constant 0 : index
// IMPERATIVE:           %[[VAL_7:.*]] = constant 16 : index
// IMPERATIVE:           %[[VAL_8:.*]] = constant 1 : index
// IMPERATIVE:           scf.for %[[VAL_9:.*]] = %[[VAL_6]] to %[[VAL_3]] step %[[VAL_8]] {
// IMPERATIVE:             scf.for %[[VAL_10:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_8]] {
// IMPERATIVE:               %[[VAL_11:.*]] = alloc() : memref<f32>
// IMPERATIVE:               store %[[VAL_5]], %[[VAL_11]][] : memref<f32>
// IMPERATIVE:               scf.for %[[VAL_12:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_8]] {
// IMPERATIVE:                 %[[VAL_13:.*]] = load %[[VAL_0]]{{\[}}%[[VAL_9]], %[[VAL_12]]] : memref<32x16xf32>
// IMPERATIVE:                 %[[VAL_14:.*]] = load %[[VAL_1]]{{\[}}%[[VAL_12]], %[[VAL_10]]] : memref<16x64xf32>
// IMPERATIVE:                 %[[VAL_15:.*]] = load %[[VAL_11]][] : memref<f32>
// IMPERATIVE:                 %[[VAL_16:.*]] = addf %[[VAL_13]], %[[VAL_14]] : f32
// IMPERATIVE:                 %[[VAL_17:.*]] = mulf %[[VAL_15]], %[[VAL_16]] : f32
// IMPERATIVE:                 store %[[VAL_17]], %[[VAL_11]][] : memref<f32>
// IMPERATIVE:               }
// IMPERATIVE:               %[[VAL_18:.*]] = load %[[VAL_11]][] : memref<f32>
// IMPERATIVE:               store %[[VAL_18]], %[[VAL_2]]{{\[}}%[[VAL_9]], %[[VAL_10]]] : memref<32x64xf32>
// IMPERATIVE:             }
// IMPERATIVE:           }
// IMPERATIVE:           return
// IMPERATIVE:         }
  // clang-format on

  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(test_conv2) {
  int64_t width = 9;
  int64_t height = 9;
  int64_t kernelWidth = 3;
  int64_t kernelHeight = 3;
  auto f32Type = FloatType::getF32(&globalContext());

  auto f = makeFunction(
      "conv2D", {},
      {MemRefType::get({height, width}, f32Type, {}, 0),
       MemRefType::get({kernelHeight, kernelWidth}, f32Type, {}, 0),
       MemRefType::get({height, width}, f32Type, {}, 0)});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  Value A = f.getArgument(0);
  Value kernel = f.getArgument(1);
  Value output = f.getArgument(2);

  makeRiseProgram(output, A, kernel)(
      [](Value A, Value kernel) { return conv2D(A, kernel); });

  std_ret();

  // generate test
  auto testFun = makeFunction("conv2D_test", {}, {});
  OpBuilder test_builder(testFun.getBody());
  ScopedContext test_scope(test_builder, testFun.getLoc());
  makeRiseTest(f, {height, width}, getFilledMemRef({height, width}),
               getFilledMemRef({kernelHeight, kernelWidth}, 1.0f));

  //  mlir::edsc::highlevel::generateTest(
  //      2, {height, width}, {kernelHeight, kernelWidth}, {height, width}, f);
  std_ret();
  // clang-format off
  // CONV_2D_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [9, 9] strides = [9, 1] data =
  // CONV_2D_TEST:       {{\[\[}}0,   1,   2,   3,   4,   5,   6,   7,   8],
  // CONV_2D_TEST:        [9,   10,   11,   12,   13,   14,   15,   16,   17],
  // CONV_2D_TEST:        [18,   19,   20,   21,   22,   23,   24,   25,   26],
  // CONV_2D_TEST:        [27,   28,   29,   30,   31,   32,   33,   34,   35],
  // CONV_2D_TEST:        [36,   37,   38,   39,   40,   41,   42,   43,   44],
  // CONV_2D_TEST:        [45,   46,   47,   48,   49,   50,   51,   52,   53],
  // CONV_2D_TEST:        [54,   55,   56,   57,   58,   59,   60,   61,   62],
  // CONV_2D_TEST:        [63,   64,   65,   66,   67,   68,   69,   70,   71],
  // CONV_2D_TEST:        [72,   73,   74,   75,   76,   77,   78,   79,   80]]
  // CONV_2D_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1] data =
  // CONV_2D_TEST:       {{\[\[}}1,   1,   1],
  // CONV_2D_TEST:        [1,   1,   1],
  // CONV_2D_TEST:        [1,   1,   1]]
  // CONV_2D_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [9, 9] strides = [9, 1] data =
  // CONV_2D_TEST:       {{\[\[}}30,   36,   45,   54,   63,   72,   81,   90,   96],
  // CONV_2D_TEST:        [84,   90,   99,   108,   117,   126,   135,   144,   150],
  // CONV_2D_TEST:        [165,   171,   180,   189,   198,   207,   216,   225,   231],
  // CONV_2D_TEST:        [246,   252,   261,   270,   279,   288,   297,   306,   312],
  // CONV_2D_TEST:        [327,   333,   342,   351,   360,   369,   378,   387,   393],
  // CONV_2D_TEST:        [408,   414,   423,   432,   441,   450,   459,   468,   474],
  // CONV_2D_TEST:        [489,   495,   504,   513,   522,   531,   540,   549,   555],
  // CONV_2D_TEST:        [570,   576,   585,   594,   603,   612,   621,   630,   636],
  // CONV_2D_TEST:        [624,   630,   639,   648,   657,   666,   675,   684,   690]]
  // clang-format on

  f.print(llvm::outs());
  testFun.print(llvm::outs());

  f.erase();
  testFun.erase();
}

//TEST_FUNC(test_conv2_separable) {
//  int64_t width = 9;
//  int64_t height = 9;
//  int64_t kernelSize = 3;
//  auto f32Type = FloatType::getF32(&globalContext());
//
//  auto f = makeFunction("conv2DSeparable", {},
//                        {MemRefType::get({height, width}, f32Type, {}, 0),
//                         MemRefType::get({kernelSize}, f32Type, {}, 0),
//                         MemRefType::get({kernelSize}, f32Type, {}, 0),
//                         MemRefType::get({height, width}, f32Type, {}, 0)});
//
//  OpBuilder builder(f.getBody());
//  ScopedContext scope(builder, f.getLoc());
//
//  Value A = f.getArgument(0);
//  Value kernelH = f.getArgument(1);
//  Value kernelV = f.getArgument(2);
//  Value output = f.getArgument(3);
//
//  makeRiseProgram(output, A, kernelH,
//                  kernelV)([](Value A, Value kernelH, Value kernelV) {
//    return conv2DSeparated(A, kernelH, kernelV, 1, 1, 1, 1);
//  });
//
//  std_ret();
//
//  // generate test
//  auto testFun = makeFunction("conv2DSeparable_test", {}, {});
//  OpBuilder test_builder(testFun.getBody());
//  ScopedContext test_scope(test_builder, testFun.getLoc());
//
//  makeRiseTest(f, {height, width}, getFilledMemRef({height, width}),
//               getFilledMemRef({3}, 1.0f), getFilledMemRef({3}, 1.0f));
//
//  std_ret();
//
//  f.print(llvm::outs());
//  testFun.print(llvm::outs());
//  f.erase();
//  testFun.erase();
//}

//
// TEST_FUNC(test_conv_tf) {
//  int64_t width = 7;
//  int64_t height = 7;
//  int64_t kernelWidth = 3;
//  int64_t kernelHeight = 3;
//  auto f32Type = FloatType::getF32(&globalContext());
//
//  auto f = makeFunction("conv2DTF", {},
//                        {MemRefType::get({1, height, width, 1}, f32Type, {},
//                        0),
//                         MemRefType::get({kernelHeight, kernelWidth, 1, 1},
//                         f32Type, {}, 0), MemRefType::get({1, height-2,
//                         width-2, 1}, f32Type, {}, 0)});
//
//  OpBuilder builder(f.getBody());
//  ScopedContext scope(builder, f.getLoc());
//
//  Value input = f.getArgument(0);
//  Value kernel = f.getArgument(1);
//  Value output = f.getArgument(2);
//
//  makeRiseProgram(output, input, kernel)([&](Value input, Value kernel){
//    return conv2DTF(input, kernel);
//  });
//  std_ret();
//
////  // generate test
//  auto testFun = makeFunction("conv2DTF_test", {}, {});
//  OpBuilder test_builder(testFun.getBody());
//  ScopedContext test_scope(test_builder, testFun.getLoc());
//  makeRiseTest(f, {1, height-2, width-2, 1}, getFilledMemRef({1, height,
//  width, 1}), getFilledMemRef({kernelHeight, kernelWidth, 1, 1}, 1.0f));
//
//  std_ret();
// clang-format off
  // CONV_2DTF_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [9, 9] strides = [9, 1] data =
  // CONV_2DTF_TEST:       {{\[\[}}1,   2,   3,   4,   5,   6,   7,   8,   9],
  // CONV_2DTF_TEST:        [10,   11,   12,   13,   14,   15,   16,   17, 18],
  // CONV_2DTF_TEST:        [19,   20,   21,   22,   23,   24,   25,   26, 27],
  // CONV_2DTF_TEST:        [28,   29,   30,   31,   32,   33,   34,   35, 36],
  // CONV_2DTF_TEST:        [37,   38,   39,   40,   41,   42,   43,   44, 45],
  // CONV_2DTF_TEST:        [46,   47,   48,   49,   50,   51,   52,   53, 54],
  // CONV_2DTF_TEST:        [55,   56,   57,   58,   59,   60,   61,   62, 63],
  // CONV_2DTF_TEST:        [64,   65,   66,   67,   68,   69,   70,   71, 72],
  // CONV_2DTF_TEST:        [73,   74,   75,   76,   77,   78,   79,   80, 81]]
  // CONV_2DTF_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1] data =
  // CONV_2DTF_TEST:       {{\[\[}}1,   1,   1],
  // CONV_2DTF_TEST:        [1,   1,   1],
  // CONV_2DTF_TEST:        [1,   1,   1]]
  // CONV_2DTF_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [9, 9] strides = [9, 1] data =
  // CONV_2DTF_TEST:       {{\[\[}}39,   45,   54,   63,   72,   81,   90,   99,105],
  // CONV_2DTF_TEST:        [93,   99,   108,   117,   126,   135,   144,   153,159],
  // CONV_2DTF_TEST:        [174,   180,   189,   198,   207,   216,   225, 234,240],
  // CONV_2DTF_TEST:        [255,   261,   270,   279,   288,   297,   306, 315,321],
  // CONV_2DTF_TEST:        [336,   342,   351,   360,   369,   378,   387, 396,402],
  // CONV_2DTF_TEST:        [417,   423,   432,   441,   450,   459,   468, 477,483],
  // CONV_2DTF_TEST:        [498,   504,   513,   522,   531,   540,   549, 558,564],
  // CONV_2DTF_TEST:        [579,   585,   594,   603,   612,   621,   630, 639,645],
  // CONV_2DTF_TEST:        [633,   639,   648,   657,   666,   675,   684, 693,699]]
// clang-format on

//
//  f.print(llvm::outs());
//  testFun.print(llvm::outs());
//  f.erase();
//  testFun.erase();
//}

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

  makeRiseProgram(output, input)([&](Value input) {
    return mlir::edsc::highlevel::stencil2D(x_size, y_size, 5, 1, 3, 1, input);
  });
  std_ret();

  // generate test
  auto testFun = makeFunction("stencil2D_test", {}, {});
  OpBuilder test_builder(testFun.getBody());
  ScopedContext test_scope(test_builder, testFun.getLoc());
  makeRiseTest(f, {x_size, y_size}, getFilledMemRef({x_size, y_size}));

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

TEST_FUNC(test_slide2d) {
  int64_t M = 7;
  int64_t N = 5;
  int slideOuter = 5;
  int slideInner = 3;
  auto f32Type = FloatType::getF32(&globalContext());

  auto f = makeFunction("slide2D", {},
                        {MemRefType::get({M, N}, f32Type, {}, 0),
                         MemRefType::get({M - 2, N - 4, slideOuter, slideInner},
                                         f32Type, {}, 0)});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  Value input = f.getArgument(0);
  Value output = f.getArgument(1);

  makeRiseProgram(output, input)([&](Value input) {
    Value slizzled = slide2D(natType(slideOuter), natType(1),
                             natType(slideInner), natType(1), input);

    return mapSeq2D(
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
  });

  std_ret();

  // generate test
  auto testFun = makeFunction("slide2D_test", {}, {});
  OpBuilder test_builder(testFun.getBody());
  ScopedContext test_scope(test_builder, testFun.getLoc());

  makeRiseTest(f, {M - 2, N - 4, slideOuter, slideInner},
               getFilledMemRef({M, N}));
  std_ret();

  f.print(llvm::outs());
  testFun.print(llvm::outs());
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

  //  Value inn = in(input, arrayType(height, arrayType(width,
  //  scalarF32Type())));

  makeRiseProgram(output, input)([&](Value input) {
    Value padded = pad2D(natType(padOuterl), natType(padOuterr),
                         natType(padInnerl), natType(padInnerr), input);

    ArrayType paddedType = padded.getType().dyn_cast<ArrayType>();
    ArrayType innerType = paddedType.getElementType().dyn_cast<ArrayType>();

    return mapSeq2D(
        scalarF32Type(),
        [&](Value elem) {
          return embed1(scalarF32Type(), elem, [&](Value elem) {
            Value cst = std_constant_float(llvm::APFloat(0.0f), f32Type);
            return elem + cst;
          });
        },
        padded);
  });

  //  out(output, mapped);

  std_ret();

  // generate test
  auto testFun = makeFunction("pad2D_test", {}, {});
  OpBuilder test_builder(testFun.getBody());
  ScopedContext test_scope(test_builder, testFun.getLoc());

  makeRiseTest(f, {outHeight, outWidth}, getFilledMemRef({height, width}));

  //  mlir::edsc::highlevel::generateTest(2, {height, width}, {outHeight,
  //  outWidth},
  //                                      f);
  std_ret();
  // clang-format off
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
  // clang-format on

  f.print(llvm::outs());
  testFun.print(llvm::outs());

  f.erase();
  testFun.erase();
}

TEST_FUNC(test_zip2D) {
  int64_t width = 3;
  int64_t height = 3;
  auto f32Type = FloatType::getF32(&globalContext());

  auto f = makeFunction("zip2D", {},
                        {MemRefType::get({height, width}, f32Type, {}, 0),
                         MemRefType::get({height, width}, f32Type, {}, 0),
                         MemRefType::get({height, width}, f32Type, {}, 0)});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  Value inputA = f.getArgument(0);
  Value inputB = f.getArgument(1);
  Value output = f.getArgument(2);

  makeRiseProgram(output, inputA, inputB)([&](Value inA, Value inB) {
    Value zipped = zip2D(inA, inB);
    return mapSeq2D(
        scalarF32Type(),
        [&](Value tuple) {
          return embed2(scalarF32Type(), ValueRange{fst(tuple), snd(tuple)},
                        [&](Value fst, Value snd) { return fst * snd; });
        },
        zipped);
  });
  std_ret();

  // generate test
  auto testFun = makeFunction("zip2D_test", {}, {});
  OpBuilder test_builder(testFun.getBody());
  ScopedContext test_scope(test_builder, testFun.getLoc());

  Value filledA = getFilledMemRef({height, width});
  Value filledB = getFilledMemRef({height, width});
  makeRiseTest(f, {height, width}, filledA, filledB);
  std_ret();

  // clang-format off
  // ZIP_2D_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1] data =
  // ZIP_2D_TEST:       {{\[\[}}0,   1,   2],
  // ZIP_2D_TEST:        [3,  4,   5],
  // ZIP_2D_TEST:        [6,  7,   8]]
  // ZIP_2D_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1] data =
  // ZIP_2D_TEST:       {{\[\[}}0,   1,   2],
  // ZIP_2D_TEST:        [3,  4,   5],
  // ZIP_2D_TEST:        [6,  7,   8]]
  // ZIP_2D_TEST:       Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [3, 3] strides = [3, 1] data =
  // ZIP_2D_TEST:       {{\[\[}}0,   1,   4],
  // ZIP_2D_TEST:        [9,   16,   25],
  // ZIP_2D_TEST:        [36,   49,   64]]
  // clang-format on

  f.print(llvm::outs());
  testFun.print(llvm::outs());

  f.erase();
  testFun.erase();
}

int main() {
  RUN_TESTS();
  return 0;
}
