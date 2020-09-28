//
// Created by martin on 26/11/2019.
//

#include "mlir/Conversion/ElevateRewriting/ElevateRewriting.h"
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
#include "mlir/EDSC/Builders.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>
#include <mlir/EDSC/Builders.h>

using namespace mlir;
using namespace mlir::rise;
using namespace mlir::edsc;

namespace {
struct ElevateRewritingPass
    : public ElevateRewritingBase<ElevateRewritingPass> {
  void runOnFunction() override;
};

struct ElevateRewritingPattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;
  LogicalResult match(FuncOp funcOp) const override;
  void rewrite(FuncOp funcOp, PatternRewriter &rewriter) const override;
};

LogicalResult ElevateRewritingPattern::match(FuncOp funcOp) const {
  bool riseInside = false;

  if (funcOp.isExternal())
    return mlir::failure();

  // Only unlowered rise programs contain RiseInOps
  funcOp.walk([&](Operation *op) {
    if (isa<InOp>(op))
      riseInside = true;
  });

  if (riseInside) {
    return success();
  } else {
    return mlir::failure();
  }
}


void ElevateRewritingPattern::rewrite(FuncOp funcOp,
                                      PatternRewriter &rewriter) const {
  OpBuilder builder(funcOp.getBody());
  ScopedContext scope(builder, funcOp.getLoc());
  // Start Elevate Rewriting from here

  Expr e;
  seq(id)(id)(e); // works

  getExpr(leftChoice(fail)(fail)(e)); // prints the expected logic error

  return;
}
} // namespace

/// gather all patterns
void mlir::populateElevatePatterns(OwningRewritePatternList &patterns,
                                   MLIRContext *ctx) {
  patterns.insert<ElevateRewritingPattern>(ctx);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// The pass:
void ElevateRewritingPass::runOnFunction() {
  auto module = getOperation();
  OwningRewritePatternList patterns;

  populateElevatePatterns(patterns, &getContext());
  ConversionTarget target(getContext());

  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addLegalDialect<rise::RiseDialect>(); // for debugging purposes

  target.addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) {
    bool riseInside = false;
    if (funcOp.isExternal())
      return true;
    funcOp.walk([&](Operation *op) {
      if (op->getDialect()->getNamespace().equals(
              rise::RiseDialect::getDialectNamespace()))
        riseInside = true;
    });
    return !riseInside;
  });

  bool erased;
  applyOpPatternsAndFold(module, patterns, &erased);
  //  applyFullConversion(module, target, patterns);
  return;
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::rise::createElevateRewritingPass() {
  return std::make_unique<ElevateRewritingPass>();
}
