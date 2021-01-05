//
// Created by martin.
//

//#include "mlir/Elevate/core.h"
#include "mlir/Elevate2/core.h"
#include "mlir/Dialect/Rise/Elevate2/traversal.h"
#include "mlir/Dialect/Rise/Elevate2/algorithmic.h"
#include "mlir/Dialect/Rise/Elevate2/predicates.h"
#include "mlir/Conversion/ElevateRewriting/ElevateRewriting.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/ElevateRewriteDriver.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::rise;
using namespace mlir::edsc;
using namespace mlir::elevate;

namespace {
struct ElevateRewritingPass
    : public ElevateRewritingBase<ElevateRewritingPass> {
  void runOnFunction() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// The pass:
void ElevateRewritingPass::runOnFunction() {
  Operation *op = getOperation();

  /////////////////////// tmp ///////////////////////
  FuncOp funcOp = dyn_cast<FuncOp>(op);
  LoweringUnitOp loweringUnit;
  funcOp.getBody().walk([&](Operation *op) {
    if (LoweringUnitOp loweringUnitOp = dyn_cast<LoweringUnitOp>(op))
      loweringUnit = loweringUnitOp;
  });
  if (!loweringUnit) {
    emitWarning(funcOp.getLoc()) << "No rise.lowering_unit found!";
    return;
  }
  // Start at the back and find the rise.out op
  Block &block = loweringUnit.region().front();
  auto _outOp = std::find_if(block.rbegin(), block.rend(),
                             [](auto &op) { return isa<OutOp>(op); });
  if (_outOp == block.rend()) {
    emitWarning(funcOp.getLoc()) << "Could not find rise.out operation!";
    return;
  }
  OutOp outOp = dyn_cast<OutOp>(*_outOp);
  auto lastApply = outOp.input().getDefiningOp();
  ///////////////////////////////////////////////////
//  bool erased;
//  applyOpPatternsAndFold(op, patterns, &erased);

  ElevateRewriteDriver rewriter(op->getContext());
  RiseDialect::dumpRiseExpression(outOp);

  RewriteResult rr_fused = topdown(fuseReduceMap())(lastApply, rewriter);
//
  auto rr_betared1 = flatMapSuccess(rr_fused,
                                    topdown(betaReduction()),
                                    rewriter);
  auto rr_betared2 = flatMapSuccess(rr_betared1, topdown(betaReduction()), rewriter);
//  lastApply = outOp.input().getDefiningOp();
//
  RiseDialect::dumpRiseExpression(outOp);
//  llvm::dbgs() << "\n\n";
//

//
//  RiseDialect::dumpRiseExpression(outOp);
//  llvm::dbgs() << "\n\n";
//
  auto rr_fissioned = normalize(mapLastFission())(lastApply,rewriter);
  lastApply = outOp.input().getDefiningOp();
  RiseDialect::dumpRiseExpression(outOp);

  auto rr_loop_blocked = seq(fmap( splitJoin(8)), splitJoin(4))(lastApply, rewriter);
  RiseDialect::dumpRiseExpression(outOp);

  lastApply = outOp.input().getDefiningOp();
//  auto rr_fissione = normalize(mapLastFission())(lastApply,rewriter);


//  RiseDialect::dumpRiseExpression(outOp);
//  funcOp.dump();
//  lastApply = outOp.input().getDefiningOp();
//  auto test = addIdAfter()(lastApply, rewriter);
//  auto rr_addid = argument(1,  argument(1, body( argument(2,addIdAfter()))))(lastApply, rewriter);
//  flatMapSuccess(rr_loop_blocked, argument(1,  seq(debug("adId"), argument(1, body( argument(2,addIdAfter()))))), rewriter);
//  lastApply = outOp.input().getDefiningOp();
//  auto rr_double_transpose = topdown(createTransposePair())(lastApply, rewriter);
//  lastApply = outOp.input().getDefiningOp();
//  RiseDialect::dumpRiseExpression(outOp);


//  auto rr_betaRed = normalize(seq(debug("betared"), betaReduction()))(lastApply, rewriter);
//  RiseDialect::dumpRiseExpression(outOp);
//  lastApply = outOp.input().getDefiningOp();
//
//  funcOp.dump();
//
//  auto rr_moved_transpose = topdown(seq(debug("move"), mapMapFBeforeTranspose()))(lastApply, rewriter);
//  lastApply = outOp.input().getDefiningOp();
  lastApply = outOp.input().getDefiningOp();

//  auto rr_fissioned2 = topdown(seq(debug("fission"), mapLastFission()))(lastApply,rewriter);

  if (auto _ = std::get_if<Failure>(&rr_fissioned)) {
    llvm::dbgs() << "logic error1!\n";
  } else {
    llvm::dbgs() << "success!\n";
  }
//  if (auto _ = std::get_if<Failure>(&rr_betared1)) {
//    llvm::dbgs() << "logic error1!\n";
//  } else {
//    llvm::dbgs() << "success!\n";
//  }
//  if (auto _ = std::get_if<Failure>(&rr_betared2)) {
//    llvm::dbgs() << "logic error1!\n";
//  } else {
//    llvm::dbgs() << "success!\n";
//  }
//  if (auto _ = std::get_if<Failure>(&rr_loop_blocked)) {
//    llvm::dbgs() << "logic error1!\n";
//  } else {
//    llvm::dbgs() << "success!\n";
//  }
//  if (auto _ = std::get_if<Failure>(&rr_addid)) {
//    llvm::dbgs() << "logic error1!\n";
//  } else {
//    llvm::dbgs() << "success!\n";
//  }
//  if (auto _ = std::get_if<Failure>(&rr_double_transpose)) {
//    llvm::dbgs() << "logic error1!\n";
//  } else {
//    llvm::dbgs() << "success!\n";
//  }
//  if (auto _ = std::get_if<Failure>(&rr_fissioned)) {
//    llvm::dbgs() << "logic error1!\n";
//  } else {
//    llvm::dbgs() << "success!\n";
//  }
//  if (auto _ = std::get_if<Failure>(&rr_moved_transpose)) {
//    llvm::dbgs() << "logic error1!\n";
//  } else {
//    llvm::dbgs() << "success!\n";
//  }
//  if (auto _ = std::get_if<Failure>(&rr_fissioned2)) {
//    llvm::dbgs() << "logic error1!\n";
//  } else {
//    llvm::dbgs() << "success!\n";
//  }
  llvm::dbgs() << "////////////// finished elevate pass! //////////////\n";
  RiseDialect::dumpRiseExpression(outOp);
  llvm::dbgs() << "\n";
  funcOp.dump();
//  RiseDialect::dumpRiseExpression2(&loweringUnit);
  return;
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::rise::createElevateRewritingPass() {
  return std::make_unique<ElevateRewritingPass>();
}

















// old pass using old system


///// gather all patterns
//void mlir::populateElevatePatterns(OwningRewritePatternList &patterns,
//                                   MLIRContext *ctx) {
//  patterns.insert<ElevateRewritingPattern>(ctx);
//
//}


//struct ElevateRewritingPattern : public OpRewritePattern<FuncOp> {
//  using OpRewritePattern<FuncOp>::OpRewritePattern;
//  LogicalResult match(FuncOp funcOp) const override;
//  void rewrite(FuncOp funcOp, PatternRewriter &rewriter) const override;
//};
//
//LogicalResult ElevateRewritingPattern::match(FuncOp funcOp) const {
//  bool riseInside = false;
//
//  if (funcOp.isExternal())
//    return mlir::failure();
//
//  // check this funcOp actually contains rise operations
//  funcOp.walk([&](Operation *op) {
//    if (isa<InOp>(op))
//      riseInside = true;
//  });
//
//  if (riseInside) {
//    return success();
//  } else {
//    return mlir::failure();
//  }
//}
//
//void ElevateRewritingPattern::rewrite(FuncOp funcOp,
//                                      PatternRewriter &rewriter) const {
//  ElevateRewriter::getInstance().rewriter = &rewriter;
//
//  // Navigating in the rise.lowering_unit to the desired ApplyOp
//  // we need for the rewrite here.
//  LoweringUnitOp loweringUnit;
//  funcOp.getBody().walk([&](Operation *op) {
//    if (LoweringUnitOp loweringUnitOp = dyn_cast<LoweringUnitOp>(op))
//      loweringUnit = loweringUnitOp;
//  });
//  if (!loweringUnit) {
//    emitError(funcOp.getLoc()) << "No rise.lowering_unit found!";
//    return;
//  }
//  // Start at the back and find the rise.out op
//  Block &block = loweringUnit.region().front();
//  auto _outOp = std::find_if(block.rbegin(), block.rend(),
//                             [](auto &op) { return isa<OutOp>(op); });
//  if (_outOp == block.rend()) {
//    emitError(funcOp.getLoc()) << "Could not find rise.out operation!";
//    return;
//  }
//  OutOp outOp = dyn_cast<OutOp>(*_outOp);
//  auto lastApply = outOp.input().getDefiningOp();
//
//  RiseDialect::dumpRiseExpression(outOp);
//
//  // clang-format off
//  // ----------- actual rewrititing starts here -----------
//  // Start Elevate Rewriting from here
//
//  bool testTraversals = true;
//  if (testTraversals) {
//    // test one
//    RewriteResult oneTestResult = one(seq(debug("test one traversal"))(fail))(*lastApply);
//    if (auto _ = std::get_if<Failure>(&oneTestResult)) {
//      llvm::dbgs() << "one: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//
//    // test function
//    RewriteResult functionTestResult = function(debug("test function traversal"))(*lastApply);
//    if (auto _ = std::get_if<Failure>(&functionTestResult)) {
//      llvm::dbgs() << "function: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//
//    // test argument
//    RewriteResult argumentTestResult = argument(1, debug("test argument traversal"))(*lastApply);
//    if (auto _ = std::get_if<Failure>(&argumentTestResult)) {
//      llvm::dbgs() << "argument: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//
//    // test fMap
//    RewriteResult fMapTestResult = fmap(debug("test fMap traversal"))(*lastApply);
//    if (auto _ = std::get_if<Failure>(&fMapTestResult)) {
//      llvm::dbgs() << "fMap: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//
//    // test body for lambda
//    RewriteResult bodyTestResult = argument(1, body(debug("test body traversal for lambda")))(*lastApply);
//    if (auto _ = std::get_if<Failure>(&argumentTestResult)) {
//      llvm::dbgs() << "body: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//
//    // test body for embed
//    RewriteResult bodyEmbedTestResult = argument(1,
//                                                 body(
//                                                     body(debug("test body traversal for embed"))
//                                                 )
//    )(*lastApply);
//    if (auto _ = std::get_if<Failure>(&bodyEmbedTestResult)) {
//      llvm::dbgs() << "bodyEmbed: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//
//    // test topdown
//    RewriteResult topdownTestResult = topdown(seq(debug("test topdown traversal"))(fail))(*lastApply);
//    if (auto _ = std::get_if<Failure>(&topdownTestResult)) {
//      llvm::dbgs() << "topdown: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//  }
//
//  bool doSplitJoin = false;
//  if (doSplitJoin) {
//    RewriteResult splitjoinResult = topdown(seq(debug("test splitjoin rewrite:"))(splitJoin(2)))(*lastApply);
//    if (auto _ = std::get_if<Failure>(&splitjoinResult)) {
//      llvm::dbgs() << "splitjoin: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//  }
//
//  // fuse reduceSeq and mapSeq
//  bool dofuseReduceMap = false;
//  if (dofuseReduceMap) {
//    RewriteResult fuseReduceMapResult = topdown(seq(debug("test fuseReduceMap rewrite:"))(fuseReduceMap))(*lastApply);
//    RewriteResult betaRed1 = topdown(betaReduction)(getExpr(fuseReduceMapResult));
//    RewriteResult betaRed2 = topdown(betaReduction)(getExpr(betaRed1));
//    lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite
//  }
//
//  bool introduceTransposePair = true;
//  if (introduceTransposePair) {
//    // test id
//    RewriteResult idTestResult = topdown(addIdAfter)(*lastApply);
//    if (auto _ = std::get_if<Failure>(&idTestResult)) {
//      llvm::dbgs() << "idTestResult: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//    lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite
//
//    // test createTransposePair
//    RewriteResult transposePairTestResult = topdown(createTransposePair)(*lastApply);
//    if (auto _ = std::get_if<Failure>(&transposePairTestResult)) {
//      llvm::dbgs() << "transposePairTestResult: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//    lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite
//
////  // test removeTransposePair
////  RewriteResult removeTransposePairTestResult = topdown(seq(debug("test removeTranposePair rewrite"))(removeTransposePair))(*lastApply);
////  if (auto _ = std::get_if<Failure>(&removeTransposePairTestResult)) {
////    llvm::dbgs() << "removeTransposePairTestResult: logic error!\n";
////  }
////  llvm::dbgs() << "\n\n";
////  lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite
//  }
//
//  auto rr = contains(outOp.getOperand(0))(*lastApply);
//  if (auto _ = std::get_if<Failure>(&rr)) {
//    llvm::dbgs() << "contains: logic error!\n";
//  }
//  llvm::dbgs() << "\n";
//
//  RiseDialect::dumpRiseExpression(outOp);
//
//  bool moveTranspose = true;
//  if (moveTranspose) {
//    // test moveTranspose
//    RewriteResult moveTransposeTestResult = topdown(seq(debug("moveTranspose"))(mapMapFBeforeTranspose))(*lastApply);
//    if (auto _ = std::get_if<Failure>(&moveTransposeTestResult)) {
//      llvm::dbgs() << "moveTransposeTestResult: logic error!\n";
//    }
//    llvm::dbgs() << "\n";
//    lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite
//  }
//
//  llvm::dbgs() << "////////////// finished elevate rewriting! //////////////";
//  // clang-format on
//  return;
//}