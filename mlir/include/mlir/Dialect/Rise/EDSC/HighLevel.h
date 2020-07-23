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

#ifndef MLIR_DIALECT_RISE_EDSC_HIGHLEVEL_H_
#define MLIR_DIALECT_RISE_EDSC_HIGHLEVEL_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Rise/EDSC/Builders.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

using namespace mlir::rise;

namespace mlir {
namespace edsc {
namespace highlevel {

void matrix_multiplication(int M, int N, int K, Value A, Value B, Value C);
Value conv2D(Value input, Value kernel);
Value conv2D(Value input, Value kernel, int padl, int padr, int padt, int padb);
Value conv2DTF(Value input, Value kernel);
void stencil(int N, int windowSize, int step, Value input, Value output);
void stencil2D(int M, int N, int outerWindowSize,
                                      int outerStep, int innerWindowSize, int innerStep,
                                      Value input, Value output);

// utilities
void generateTest(int dims, ArrayRef<int64_t> inSizes,
                  ArrayRef<int64_t> outSizes, FuncOp riseFun = nullptr);
void generateTest(int dims, ArrayRef<int64_t> inSizesA, ArrayRef<int64_t> inSizesB,
                                         ArrayRef<int64_t> outSizes,
                                         FuncOp riseFun = nullptr);
} // namespace highlevel
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_RISE_EDSC_HIGHLEVEL_H_
