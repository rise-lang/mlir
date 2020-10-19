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

#include "mlir/Elevate/ElevateRewriter.h"

#include <iostream>
#include <mlir/EDSC/Builders.h>

using namespace mlir;
using namespace mlir::rise;
using namespace mlir::edsc;
using namespace mlir::elevate;

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

  ElevateRewriter::getInstance().rewriter = &rewriter;


  LoweringUnitOp loweringUnit;
  funcOp.getBody().walk([&](Operation *op) {
    if (LoweringUnitOp loweringUnitOp = dyn_cast<LoweringUnitOp>(op))
      loweringUnit = loweringUnitOp;
  });
  if (!loweringUnit) {
    emitError(funcOp.getLoc()) << "No rise.lowering_unit found!";
    return;
  }
  // Start at the back and find the rise.out op
  Block &block = loweringUnit.region().front();
  auto _outOp = std::find_if(block.rbegin(), block.rend(),
                             [](auto &op) { return isa<OutOp>(op); });
  if (_outOp == block.rend()) {
    emitError(funcOp.getLoc()) << "Could not find rise.out operation!";
    return;
  }
  OutOp outOp = dyn_cast<OutOp>(*_outOp);
  auto lastApply = outOp.input().getDefiningOp();

  // clang-format off
  // Start Elevate Rewriting from here

  // test one
  RewriteResult oneTestResult = one(seq(debug("one"))(fail))(*lastApply);
  if (auto _ = std::get_if<Failure>(&oneTestResult)) {
    std::cout << "one: logic error!\n" << std::flush;
  }
  std::cout << "\n\n" << std::flush;

  // test function
  RewriteResult functionTestResult = function(debug("function"))(*lastApply);
  if (auto _ = std::get_if<Failure>(&functionTestResult)) {
    std::cout << "function: logic error!\n" << std::flush;
  }
  std::cout << "\n\n" << std::flush;

  // test argument
  RewriteResult argumentTestResult = argument(1, debug("argument"))(*lastApply);
  if (auto _ = std::get_if<Failure>(&argumentTestResult)) {
    std::cout << "argument: logic error!\n" << std::flush;
  }
  std::cout << "\n\n" << std::flush;

  // test fMap
  RewriteResult fMapTestResult = fmap(debug("fMap"))(*lastApply);
  if (auto _ = std::get_if<Failure>(&fMapTestResult)) {
    std::cout << "fMap: logic error!\n" << std::flush;
  }
  std::cout << "\n\n" << std::flush;

  // test body for lambda
  RewriteResult bodyTestResult = argument(1, body(debug("body")))(*lastApply);
  if (auto _ = std::get_if<Failure>(&argumentTestResult)) {
    std::cout << "body: logic error!\n" << std::flush;
  }
  std::cout << "\n\n" << std::flush;

  // test body for embed
  RewriteResult bodyEmbedTestResult = argument(1,
                                               body(
                                                   seq(
                                                       debug("body"))(
                                                       body(debug("body"))
                                                       )
                                                   )
                                               )(*lastApply);
  if (auto _ = std::get_if<Failure>(&bodyEmbedTestResult)) {
    std::cout << "bodyEmbed: logic error!\n" << std::flush;
  }
  std::cout << "\n\n" << std::flush;

  // test topdown
  RewriteResult topdownTestResult = topdown(seq(debug("topdown"))(fail))(*lastApply);
  if (auto _ = std::get_if<Failure>(&topdownTestResult)) {
    std::cout << "topdown: logic error!\n" << std::flush;
  }
  std::cout << "\n\n" << std::flush;



  bool doSplitJoin = true;
  if (doSplitJoin) {
    RewriteResult splitjoinResult = topdown(seq(debug("splitjoin:"))(splitJoin(2)))(*lastApply);
    if (auto _ = std::get_if<Failure>(&splitjoinResult)) {
      std::cout << "splitjoin: logic error!\n" << std::flush;
    }
    std::cout << "\n\n" << std::flush;
  }

  // fuse reduceSeq and mapSeq
  bool dofuseReduceMap = true;
  if (dofuseReduceMap) {
    RewriteResult fuseReduceMapResult = topdown(seq(debug("fuseReduceMap:"))(fuseReduceMap))(*lastApply);
    RewriteResult betaRed1 = topdown(betaReduction)(getExpr(fuseReduceMapResult));
    RewriteResult betaRed2 = topdown(betaReduction)(getExpr(betaRed1));
  }
  lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite

  // test id
  RewriteResult idTestResult = topdown(seq(debug("id"))(addIdAfter))(*lastApply);
  if (auto _ = std::get_if<Failure>(&idTestResult)) {
    std::cout << "idTestResult: logic error!\n" << std::flush;
  }
  std::cout << "\n\n" << std::flush;
  lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite

  // test createTransposePair
  RewriteResult transposePairTestResult = topdown(seq(debug("tranposePair"))(createTransposePair))(*lastApply);
  if (auto _ = std::get_if<Failure>(&transposePairTestResult)) {
    std::cout << "transposePairTestResult: logic error!\n" << std::flush;
  }
  std::cout << "\n\n" << std::flush;
  lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite

  // test removeTransposePair
  RewriteResult removeTransposePairTestResult = topdown(seq(debug("removeTranposePair"))(removeTransposePair))(*lastApply);
  if (auto _ = std::get_if<Failure>(&removeTransposePairTestResult)) {
    std::cout << "removeTransposePairTestResult: logic error!\n" << std::flush;
  }
  std::cout << "\n\n" << std::flush;
  lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite

  // test createTransposePair
//  RewriteResult moveTransposeTestResult = topdown(seq(debug("moveTranspose"))(transposeBeforeMapMap))(*lastApply);
//  if (auto _ = std::get_if<Failure>(&moveTransposeTestResult)) {
//    std::cout << "moveTransposeTestResult: logic error!\n" << std::flush;
//  }
//  std::cout << "\n\n" << std::flush;
//  lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite


  std::cout << "\n\n" << std::flush;
  std::cout << "///////////////////// finished rewriting! /////////////////////\n\n\n";
  // clang-format on
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
