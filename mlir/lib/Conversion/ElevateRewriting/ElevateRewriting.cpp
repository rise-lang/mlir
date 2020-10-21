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

// This pattern obviously will not converge. It is not meant to and is only used
// for prototyping purposes. This will eventually change to another
// infrastructure, probably replacing GreedyPatternRewriteDriver.

struct ElevateRewritingPattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;
  LogicalResult match(FuncOp funcOp) const override;
  void rewrite(FuncOp funcOp, PatternRewriter &rewriter) const override;
};

LogicalResult ElevateRewritingPattern::match(FuncOp funcOp) const {
  bool riseInside = false;

  if (funcOp.isExternal())
    return mlir::failure();

  // check this funcOp actually contains rise operations
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
//  ElevateRewriter::getInstance().rewriter = &rewriter;


  // Navigating in the rise.lowering_unit to the desired ApplyOp
  // we need for the rewrite here.
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

  // for reproducing immediately get applyOp needed for the rewrite:
  auto expr = dyn_cast<ApplyOp>(lastApply).getOperand(1).getDefiningOp();



  // clang-format off
  // ----------- actual rewrititing starts here -----------
  // Start Elevate Rewriting from here

  bool testTraversals = false;
  if (testTraversals) {
    // test one
    RewriteResult oneTestResult = one(seq(debug("test one traversal"))(fail))(*lastApply);
    if (auto _ = std::get_if<Failure>(&oneTestResult)) {
      llvm::dbgs() << "one: logic error!\n";
    }
    llvm::dbgs() << "\n";

    // test function
    RewriteResult functionTestResult = function(debug("test function traversal"))(*lastApply);
    if (auto _ = std::get_if<Failure>(&functionTestResult)) {
      llvm::dbgs() << "function: logic error!\n";
    }
    llvm::dbgs() << "\n";

    // test argument
    RewriteResult argumentTestResult = argument(1, debug("test argument traversal"))(*lastApply);
    if (auto _ = std::get_if<Failure>(&argumentTestResult)) {
      llvm::dbgs() << "argument: logic error!\n";
    }
    llvm::dbgs() << "\n";

    // test fMap
    RewriteResult fMapTestResult = fmap(debug("test fMap traversal"))(*lastApply);
    if (auto _ = std::get_if<Failure>(&fMapTestResult)) {
      llvm::dbgs() << "fMap: logic error!\n";
    }
    llvm::dbgs() << "\n";

    // test body for lambda
    RewriteResult bodyTestResult = argument(1, body(debug("test body traversal for lambda")))(*lastApply);
    if (auto _ = std::get_if<Failure>(&argumentTestResult)) {
      llvm::dbgs() << "body: logic error!\n";
    }
    llvm::dbgs() << "\n";

    // test body for embed
    RewriteResult bodyEmbedTestResult = argument(1,
                                                 body(
                                                     body(debug("test body traversal for embed"))
                                                 )
    )(*lastApply);
    if (auto _ = std::get_if<Failure>(&bodyEmbedTestResult)) {
      llvm::dbgs() << "bodyEmbed: logic error!\n";
    }
    llvm::dbgs() << "\n";

    // test topdown
    RewriteResult topdownTestResult = topdown(seq(debug("test topdown traversal"))(fail))(*lastApply);
    if (auto _ = std::get_if<Failure>(&topdownTestResult)) {
      llvm::dbgs() << "topdown: logic error!\n";
    }
    llvm::dbgs() << "\n";
  }

  bool doSplitJoin = false;
  if (doSplitJoin) {
    RewriteResult splitjoinResult = topdown(seq(debug("test splitjoin rewrite:"))(splitJoin(2)))(*lastApply);
    if (auto _ = std::get_if<Failure>(&splitjoinResult)) {
      llvm::dbgs() << "splitjoin: logic error!\n";
    }
    llvm::dbgs() << "\n";
  }

  // fuse reduceSeq and mapSeq
  bool dofuseReduceMap = false;
  if (dofuseReduceMap) {
    RewriteResult fuseReduceMapResult = topdown(seq(debug("test fuseReduceMap rewrite:"))(fuseReduceMap))(*lastApply);
    RewriteResult betaRed1 = topdown(betaReduction)(getExpr(fuseReduceMapResult));
    RewriteResult betaRed2 = topdown(betaReduction)(getExpr(betaRed1));
    lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite
  }

  bool introduceTransposePair = false;
  if (introduceTransposePair) {
    // test id
    RewriteResult idTestResult = topdown(addIdAfter)(*lastApply);
    if (auto _ = std::get_if<Failure>(&idTestResult)) {
      llvm::dbgs() << "idTestResult: logic error!\n";
    }
    llvm::dbgs() << "\n";
    lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite

    // test createTransposePair
    RewriteResult transposePairTestResult = topdown(createTransposePair)(*lastApply);
    if (auto _ = std::get_if<Failure>(&transposePairTestResult)) {
      llvm::dbgs() << "transposePairTestResult: logic error!\n";
    }
    llvm::dbgs() << "\n";
    lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite

//  // test removeTransposePair
//  RewriteResult removeTransposePairTestResult = topdown(seq(debug("test removeTranposePair rewrite"))(removeTransposePair))(*lastApply);
//  if (auto _ = std::get_if<Failure>(&removeTransposePairTestResult)) {
//    llvm::dbgs() << "removeTransposePairTestResult: logic error!\n";
//  }
//  llvm::dbgs() << "\n\n";
//  lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite
  }

  bool moveTranspose = false;
  if (moveTranspose) {
    // test moveTranspose
    RewriteResult moveTransposeTestResult = topdown(seq(debug("hi"))(transposeBeforeMapMap))(*lastApply);
    if (auto _ = std::get_if<Failure>(&moveTransposeTestResult)) {
      llvm::dbgs() << "moveTransposeTestResult: logic error!\n";
    }
    llvm::dbgs() << "\n";
    lastApply = outOp.input().getDefiningOp();  // has been changed by this rewrite
  }

  bool rewriteWithoutElevate = true;
  if (rewriteWithoutElevate) {
    // Check whether the computation is structured correctly and get references
    // to all required Values
    if (!isa<ApplyOp>(expr)) return;
    auto apply1 = cast<ApplyOp>(expr);
    if (!isa<mlir::rise::TransposeOp>(apply1.getOperand(0).getDefiningOp())) return;
    auto transpose1 = cast<mlir::rise::TransposeOp>(apply1.getOperand(0).getDefiningOp());
    if (!apply1.getOperand(1).isa<OpResult>()) return;
    if (!isa<ApplyOp>(apply1.getOperand(1).getDefiningOp())) return;
    auto apply2 = cast<ApplyOp>(apply1.getOperand(1).getDefiningOp());
    if (!isa<MapSeqOp>(apply2.getOperand(0).getDefiningOp())) return;
    auto mapSeqOp1 = cast<MapSeqOp>(apply2.getOperand(0).getDefiningOp());
    if (!apply2.getOperand(1).isa<OpResult>()) return;
    if (!isa<LambdaOp>(apply2.getOperand(1).getDefiningOp())) return;
    auto outerMapLambda = cast<LambdaOp>(apply2.getOperand(1).getDefiningOp()); // %2
    if (!isa<ApplyOp>(outerMapLambda.region().front().getTerminator()->getOperand(0).getDefiningOp())) return;
    auto apply3 = cast<ApplyOp>(outerMapLambda.region().front().getTerminator()->getOperand(0).getDefiningOp());
    if (!isa<MapSeqOp>(apply3.getOperand(0).getDefiningOp())) return;
    auto mapSeqOp2 = cast<MapSeqOp>(apply3.getOperand(0).getDefiningOp());
    if (!isa<LambdaOp>(apply3.getOperand(1).getDefiningOp())) return;
    auto f = cast<LambdaOp>(apply3.getOperand(1).getDefiningOp()); // %9
    // successful match

    ScopedContext scope(rewriter, expr->getLoc());
    rewriter.setInsertionPointAfter(apply1);

//    apply1.getParentOfType<FuncOp>().dump();

    Operation *lambdaCopy = rewriter.clone(*f);

//    apply1.getParentOfType<FuncOp>().dump();

    Value result = mapSeq("affine", mapSeqOp1.t(), [&](Value elem){
      return mapSeq("affine", mapSeqOp2.t(), lambdaCopy->getResult(0), apply3.getOperand(2));
    }, transpose(apply2.getOperand(2)));

    // Skipping the computation and just replacing the value also produces the error.
    // So the error should not be due to the usage of edsc
//    apply1.replaceAllUsesWith(apply2.getOperand(2).getDefiningOp());


    // cleanup
    apply1.replaceAllUsesWith(result.getDefiningOp());

    rewriter.eraseOp(apply1);
    rewriter.eraseOp(transpose1);
    rewriter.eraseOp(apply2);
    rewriter.eraseOp(mapSeqOp1);

    // TODO: for some reason outerMapLambda still has uses (the only use was by
    //  apply2, which should be erased now)
    //  I am leaving it out for now, which triggers an error later in the verifier
//    rewriter.eraseOp(outerMapLambda);
  }

  llvm::dbgs() << "////////////// finished elevate rewriting! //////////////";
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
  target.addLegalDialect<rise::RiseDialect>();

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
  return;
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::rise::createElevateRewritingPass() {
  return std::make_unique<ElevateRewritingPass>();
}
