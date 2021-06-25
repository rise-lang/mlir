//
// Created by martin on 26/11/2019.
//

#include "mlir/Conversion/RiseBuilderTest/RiseBuilderTest.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>
#include <mlir/EDSC/Builders.h>

using namespace mlir;
using namespace mlir::rise;
using namespace mlir::edsc;

namespace {
struct RiseBuilderTestPass
    : public RiseBuilderTestBase<RiseBuilderTestPass> {
  void runOnFunction() override;
};

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct RiseBuilderTestPattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;
  LogicalResult match(FuncOp funcOp) const override;
  void rewrite(FuncOp funcOp, PatternRewriter &rewriter) const override;
};

LogicalResult RiseBuilderTestPattern::match(FuncOp funcOp) const {
  // always match. I set the number of iterations of pattern applications to 1,
  // so the pattern is only applied once
  return success();
}

void RiseBuilderTestPattern::rewrite(FuncOp funcOp,
                                      PatternRewriter &b) const {
  //===--------------------------------------------------------------------===//
  // rewriting here
  //===--------------------------------------------------------------------===//
  ///
  /// PatternRewriter inherits from OpBuilder.
  /// lets always use b.create so it is adequate everywhere
  /// Furthermore, let's use loc everywhere for the location

  // some placeholder location
  Location loc = b.getUnknownLoc();


  // creating inOps for the funcOpArgs
  rise::ScalarType scalarF32Type = rise::ScalarType::get(b.getContext(), Float32Type::get(b.getContext()));
  rise::ArrayType inAType = rise::ArrayType::get(b.getContext(), rise::Nat::get(b.getContext(), 1024), scalarF32Type);

  b.setInsertionPointToStart(&funcOp.getBody().front());
  LoweringUnitOp loweringUnit = b.create<LoweringUnitOp>(loc);
  // setting insertion point to beginning of block:
  b.setInsertionPointToStart(loweringUnit.getBody());

  // emit two  rise.in ops
  rise::InOp inAOp = b.create<InOp>(loc, TypeRange{inAType}, funcOp.getArgument(1));
  rise::InOp inBOp = b.create<InOp>(loc, TypeRange{inAType}, funcOp.getArgument(2));


  // The painful thing doing everything this way is that everything is very explicit and mistakes here can yield invalid IR.
  // I could invest into improving the builders (i.e. b.create<>) to infer more information than they do now.
  // Examples for how exactly to construct the types are to be found in mlir/Dialect/Rise/EDSC/HighLevel.cpp
  // Look at the base case for e.g. zip to find the code.

  // another useful tip: ctrl + click on ZipOp after compiling at least once to see the available build calls

  rise::Nat n = rise::Nat::get(b.getContext(), 1024);
  rise::FunType zipType = rise::FunType::get(
      b.getContext(), rise::ArrayType::get(b.getContext(), n, scalarF32Type),
      rise::FunType::get(b.getContext(), rise::ArrayType::get(b.getContext(), n, scalarF32Type),
                         rise::ArrayType::get(b.getContext(), n, rise::Tuple::get(b.getContext(), scalarF32Type, scalarF32Type))));
  rise::ZipOp zipOp =
      b.create<ZipOp>(loc, zipType, rise::NatAttr::get(b.getContext(), n),
                      rise::DataTypeAttr::get(b.getContext(), scalarF32Type),
                      rise::DataTypeAttr::get(b.getContext(), scalarF32Type));


  // TODO: Did not finish before leaving:
  //  generate MapSeq(mul)
  // generate outOp

  return;
}
} // namespace

/// gather all patterns
void mlir::populateRiseBuilderTestPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<RiseBuilderTestPattern>(ctx);
}
//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// The pass:
void RiseBuilderTestPass::runOnFunction() {
  auto func = getOperation();
  RewritePatternSet patterns(func->getContext());
  populateRiseBuilderTestPatterns(patterns, &getContext());

  bool erased;
  applyOpPatternsAndFold(func, std::move(patterns), &erased);
  return;
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::rise::createRiseBuilderTestPass() {
  return std::make_unique<RiseBuilderTestPass>();
}
