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
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

using namespace mlir::rise;

namespace mlir {
namespace edsc {
namespace highlevel {

void matrix_multiplication(int M, int N, int K, Value A, Value B, Value C);
void convolution();
void stencil(int n, Value input, Value output);
void stencil2D(int n, Value input, Value output);

// utilities
void generateTest(int dims, ArrayRef<int64_t> inSizes,
                  ArrayRef<int64_t> outSizes, FuncOp riseFun = nullptr);

} // namespace highlevel
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_RISE_EDSC_HIGHLEVEL_H_
