//
// Created by martin on 13/04/2020.
//

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>

using namespace mlir;
using namespace mlir::rise;

namespace {
struct RiseRewritingPass : public ModulePass<RiseRewritingPass> {
  void runOnModule() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//
using OpListType = llvm::iplist<Operation>;

struct mmRewriting : public RewritePattern {
  template <typename OpT>
  OpT findOp(Block &block, OpListType::iterator iterator) const;

  mmRewriting(MLIRContext *context)
      : RewritePattern(RiseFunOp::getOperationName(), 0, context) {}

  // In this function match for a mm computation and save in the
  // PatternMatchState where it is and rewrite it below.
  PatternMatchResult match(Operation *op) const override {
    auto riseFunOp = dyn_cast<RiseFunOp>(op);

    auto outerLambda =
        findOp<LambdaOp>(riseFunOp.region().front(),
                         riseFunOp.region().front().getOperations().begin());
    if (!outerLambda)
      return matchFailure();

    auto innerLambda =
        findOp<LambdaOp>(outerLambda.region().front(),
                         outerLambda.region().front().getOperations().begin());
    if (!innerLambda)
      return matchFailure();

    auto zipOp =
        findOp<ZipOp>(innerLambda.region().front(),
                      innerLambda.region().front().getOperations().begin());
    if (!zipOp)
      return matchFailure();

    auto zipApply = findOp<ApplyOp>(innerLambda.region().front(),
                                    OpListType::iterator(zipOp));
    if (!zipApply)
      return matchFailure();

    auto reductionLambda =
        findOp<LambdaOp>(innerLambda.region().front(),
                         innerLambda.region().front().getOperations().begin());
    if (!reductionLambda)
      return matchFailure();

    // Checking for all operations inside ReductionLambda
    if (!findOp<FstOp>(
            reductionLambda.region().front(),
            reductionLambda.region().front().getOperations().begin()))
      return matchFailure();
    if (!findOp<SndOp>(
            reductionLambda.region().front(),
            reductionLambda.region().front().getOperations().begin()))
      return matchFailure();
    if (!findOp<MulFOp>(
            reductionLambda.region().front(),
            reductionLambda.region().front().getOperations().begin()))
      return matchFailure();
    if (!findOp<AddFOp>(
            reductionLambda.region().front(),
            reductionLambda.region().front().getOperations().begin()))
      return matchFailure();

    auto literalOp =
        findOp<LiteralOp>(innerLambda.region().front(),
                          innerLambda.region().front().getOperations().begin());
    if (!literalOp)
      return matchFailure();

    auto reduceOp = findOp<ReduceSeqOp>(
        innerLambda.region().front(),
        innerLambda.region().front().getOperations().begin());
    if (!reduceOp)
      return matchFailure();
    auto reduceApply = findOp<ApplyOp>(innerLambda.region().front(),
                                       OpListType::iterator(reduceOp));
    if (!reduceApply)
      return matchFailure();

    if (!reduceApply.getOperand(0) == reduceOp ||
        !reduceApply.getOperand(1) == reductionLambda ||
        !reduceApply.getOperand(2) == zipApply)
      return matchFailure();

    auto innerMap =
        findOp<MapSeqOp>(outerLambda.region().front(),
                         outerLambda.region().front().getOperations().begin());
    if (!innerMap)
      return matchFailure();

    auto innerMapApply = findOp<ApplyOp>(outerLambda.region().front(),
                                         OpListType::iterator(innerMap));
    if (!innerMapApply)
      return matchFailure();

    if (!innerMapApply.getOperand(0) == innerMap ||
        !innerMapApply.getOperand(1) == innerLambda)
      return matchFailure();

    auto outerMap =
        findOp<MapSeqOp>(riseFunOp.region().front(),
                         riseFunOp.region().front().getOperations().begin());
    if (!outerMap)
      return matchFailure();

    auto outerMapApply = findOp<ApplyOp>(riseFunOp.region().front(),
                                         OpListType::iterator(outerMap));
    if (!outerMapApply)
      return matchFailure();

    if (!outerMapApply.getOperand(0) == outerMap ||
        !outerMapApply.getOperand(1) == outerLambda)
      return matchFailure();

    return matchSuccess();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    std::cout << "\n\nrewriting! \n\n" << std::flush;
    if (isa<RiseFunOp>(op)) {
      rewriter.eraseOp(op);
      // generate unwraps
      //      rewriter.create<linalg::MatmulOp>(rewriter.getContext(), );
      // generate wraps
    }
    return;
  }
};

template <typename OpT>
OpT mmRewriting::findOp(Block &block, OpListType::iterator iterator) const {
  for (OpListType::iterator it = iterator; it != block.getOperations().end();
       it++) {
    if (OpT foundOp = dyn_cast<OpT>(*it))
      return foundOp;
  }
  return nullptr;
}
void RiseRewritingPass::runOnModule() {
  auto *context = &this->getContext();
  OwningRewritePatternList patterns;
  patterns.insert<mmRewriting>(context);

  applyPatternsGreedily(this->getOperation(), patterns);
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::rise::createRiseRewritingPass() {
  return std::make_unique<RiseRewritingPass>();
}

static PassRegistration<RiseRewritingPass>
    pass("rise-rewriting",
         "Some informative description about these rewrites.");
