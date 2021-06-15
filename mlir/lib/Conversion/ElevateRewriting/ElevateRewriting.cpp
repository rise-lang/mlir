//
// Created by martin.
//

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
// Strategies
//===----------------------------------------------------------------------===//

RewriteResult matmulBaseline(Operation *op, ElevateRewriteDriver &rewriter) {
  RewriteResult rr = topdown(fuseReduceMap())(op, rewriter);
  rr = flatMapSuccess(rr, normalize(topdown(betaReduction())), rewriter);

  // complete strategy:
  // auto matmulBaseline = seq(topdown(fuseReduceMap()), normalize(topdown(betaReduction())));
  return rr;
}

RewriteResult matmulTile(Operation *op, FuncOp func, ElevateRewriteDriver &rewriter) {
  // baseline
  RewriteResult rr = topdown(fuseReduceMap())(op, rewriter);
  rr = flatMapSuccess(rr, normalize(betaReduction()), rewriter);

  // blocking
  // tile(8,4)
  rr = seq(fmap( splitJoin(4)), splitJoin(8))(getLastApply(func), rewriter);

  // interchange
  rr = flatMapSuccess(rr, normalize(mapLastFission()), rewriter);
  rr = flatMapSuccess(rr, argument(argument(fmap(addIdAfter()))), rewriter);
  rr = flatMapSuccess(rr, topdown(createTransposePair()), rewriter);
  rr = flatMapSuccess(rr, topdown(moveMapMapFBeforeTranspose()), rewriter);

  rr = flatMapSuccess(rr, normalize(mapLastFission()), rewriter);

  // TODO: compose complete strategy in one go
  return rr;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// The pass:
void ElevateRewritingPass::runOnFunction() {
  FuncOp func = getOperation();
  ElevateRewriteDriver rewriter(func.getContext());
  auto lastApply = getLastApply(func);
  if (!lastApply) return;
  llvm::dbgs() << "initial expression:\n";
  RiseDialect::dumpRiseExpression(lastApply);

  auto rr =  elevate::id()(lastApply, rewriter);

//  rr = matmulBaseline(lastApply, rewriter);
  rr = matmulTile(lastApply, func, rewriter);


  if (std::get_if<Success>(&rr)) {
    llvm::dbgs() << "\nsuccess\n";
  } else {
    llvm::dbgs() << "\nfailure\n";
  }

  llvm::dbgs() << "\n////////////// finished elevate pass! //////////////\n\n";
  func->dump();
  RiseDialect::dumpRiseExpression(getLastApply(func));
  llvm::dbgs() << "\n";
  return;
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::rise::createElevateRewritingPass() {
  return std::make_unique<ElevateRewritingPass>();
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

ApplyOp mlir::rise::getLastApply(FuncOp funcOp) {
  LoweringUnitOp loweringUnit;
  funcOp.getBody().walk([&](Operation *op) {
    if (LoweringUnitOp loweringUnitOp = dyn_cast<LoweringUnitOp>(op))
      loweringUnit = loweringUnitOp;
  });
  if (!loweringUnit) {
    emitWarning(funcOp.getLoc()) << "No rise.lowering_unit found!";
    return nullptr;
  }
  // Start at the back and find the rise.out op
  Block &block = loweringUnit.region().front();
  auto _outOp = std::find_if(block.rbegin(), block.rend(),
                             [](auto &op) { return isa<OutOp>(op); });
  if (_outOp == block.rend()) {
    emitWarning(funcOp.getLoc()) << "Could not find rise.out operation!";
    return nullptr;
  }
  OutOp outOp = dyn_cast<OutOp>(*_outOp);
  auto lastApply = outOp.input().getDefiningOp();

  return dyn_cast<ApplyOp>(lastApply);
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