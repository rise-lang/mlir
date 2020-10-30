//===- GreedyPatternRewriteDriver.cpp - A greedy rewriter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mlir::applyPatternsAndFoldGreedily.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/ElevateRewriteDriver.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Elevate2/core.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "elevate-pattern-matcher"


//===----------------------------------------------------------------------===//
// ElevateRewriteDriver
//===----------------------------------------------------------------------===//
mlir::ElevateRewriteDriver::ElevateRewriteDriver(MLIRContext *ctx) : PatternRewriter(ctx) {
//    ElevateRewriter::getInstance().rewriter = this;
//
//  auto idRewrite = mlir::elevate2::IdRewritePattern();
//  auto failRewrite = mlir::elevate2::FailRewritePattern();
//
//  mlir::elevate2::id = mlir::elevate2::Strategy(&idRewrite);
//  mlir::elevate2::fail = mlir::elevate2::Strategy(&failRewrite);
}

//mlir::elevate2::RewriteResult mlir::ElevateRewriteDriver::rewrite(Operation *op, mlir::elevate2::Strategy strategy) {
//  mlir::elevate2::RewriteResult rr = strategy(*op);
//
//  return rr;
//}

void mlir::ElevateRewriteDriver::notifyOperationInserted(Operation *op) {};

void mlir::ElevateRewriteDriver::notifyOperationRemoved(Operation *op) {};

void mlir::ElevateRewriteDriver::notifyRootReplaced(Operation *op) {
    llvm::dbgs() << "root replaced!\n";

};
