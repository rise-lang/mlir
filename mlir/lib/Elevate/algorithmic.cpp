//
// Created by martin on 30/10/2020.
//

#include "mlir/Dialect/Rise/Elevate2/algorithmic.h"

using namespace mlir::rise;
using namespace mlir::edsc::abstraction;
using namespace mlir::edsc::op;
using namespace mlir::edsc::type;
using namespace mlir::elevate;

RewriteResult mlir::elevate::SplitJoinRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto applyMap = cast<ApplyOp>(op);

  if (!isa<MapSeqOp>(applyMap.getOperand(0).getDefiningOp())) return Failure();
  auto mapSeqOp = cast<MapSeqOp>(applyMap.getOperand(0).getDefiningOp());
  if (mapSeqOp.n().getIntValue() % n != 0) return Failure();
  // successful match

  auto mapLambda = applyMap.getOperand(1).getDefiningOp();
  Value mapInput = applyMap.getOperand(2);

  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(applyMap);

  Value result = join(mapSeq2D(mapSeqOp.t(), mapLambda->getResult(0), split(natType(n), mapInput)));

  return success(result.getDefiningOp());
}
auto mlir::elevate::splitJoin(const int n) -> SplitJoinRewritePattern {
  return SplitJoinRewritePattern(n);
}

RewriteResult mlir::elevate::FuseReduceMapRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto applyReduction = cast<ApplyOp>(op);

  if (!isa<ReduceSeqOp>(applyReduction.getOperand(0).getDefiningOp())) return Failure();
  auto reduction = cast<ReduceSeqOp>(applyReduction.getOperand(0).getDefiningOp());
  auto reductionLambda = applyReduction.getOperand(1).getDefiningOp();
  auto initializer = applyReduction.getOperand(2).getDefiningOp();

  if (!isa<ApplyOp>(applyReduction.getOperand(3).getDefiningOp())) return Failure();
  auto reductionInput = cast<ApplyOp>(applyReduction.getOperand(3).getDefiningOp());

  if (!isa<MapSeqOp>(reductionInput.getOperand(0).getDefiningOp())) return Failure();
  // successful match
  auto mapSeq = cast<MapSeqOp>(reductionInput.getOperand(0).getDefiningOp());
  auto mapLambda = reductionInput.getOperand(1).getDefiningOp();
  Value mapInput = reductionInput.getOperand(2);

  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(applyReduction);

  Value newReduceApplication = reduceSeq(scalarF32Type(), [&](Value y, Value acc){
    Value mapped = apply(scalarF32Type(), mapLambda->getResult(0), y);
    return apply(scalarF32Type(), reductionLambda->getResult(0), {mapped, acc});
  },initializer->getResult(0), mapInput);

  Operation *result = newReduceApplication.getDefiningOp();
  return success(result);
}
auto mlir::elevate::fuseReduceMap() -> FuseReduceMapRewritePattern { return FuseReduceMapRewritePattern(); }

RewriteResult mlir::elevate::BetaReductionRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply = cast<ApplyOp>(op);
  if (!isa<LambdaOp>(apply.getOperand(0).getDefiningOp())) return Failure();
  // match success
  auto lambda = cast<LambdaOp>(apply.getOperand(0).getDefiningOp());

  SmallVector<Value, 10> args = SmallVector<Value, 10>();
  for (int i = 1; i < apply.getNumOperands(); i++) {
    args.push_back(apply.getOperand(i));
  }
  substitute(lambda, args);
  Value inlinedLambdaResult = inlineLambda(lambda, op->getBlock(), apply);
  Operation *result = inlinedLambdaResult.getDefiningOp();

  return success(result);
}
auto mlir::elevate::betaReduction() -> BetaReductionRewritePattern { return BetaReductionRewritePattern(); }

RewriteResult mlir::elevate::AddIdAfterRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply = cast<ApplyOp>(op);

  // successful match
  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(apply);

  auto newApply = rewriter.clone(*apply);
  Value result = mlir::edsc::op::id(newApply->getResult(0));

  return success(result.getDefiningOp());
}
auto mlir::elevate::addIdAfter() -> AddIdAfterRewritePattern { return AddIdAfterRewritePattern(); }

RewriteResult mlir::elevate::CreateTransposePairRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply = cast<ApplyOp>(op);
  if (!isa<IdOp>(apply.getOperand(0).getDefiningOp())) return Failure();
  auto id = cast<IdOp>(apply.getOperand(0).getDefiningOp());
  if (!apply.getResult().getType().isa<ArrayType>()) return Failure();
  if (!apply.getResult().getType().dyn_cast<ArrayType>().getElementType().isa<ArrayType>()) return Failure();

  // successful match
  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(apply);
  Value result = transpose(transpose(apply.getOperand(1)));

  return success(result.getDefiningOp());
}
auto mlir::elevate::createTransposePair() -> CreateTransposePairRewritePattern { return CreateTransposePairRewritePattern(); }

RewriteResult mlir::elevate::RemoveTransposePairRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply1 = cast<ApplyOp>(op);
  if (!isa<mlir::rise::TransposeOp>(apply1.getOperand(0).getDefiningOp())) return Failure();
  auto transpose1 = cast<mlir::rise::TransposeOp>(apply1.getOperand(0).getDefiningOp());
  if (!apply1.getOperand(1).isa<OpResult>()) return Failure();
  if (!isa<ApplyOp>(apply1.getOperand(1).getDefiningOp())) return Failure();
  auto apply2 = cast<ApplyOp>(apply1.getOperand(1).getDefiningOp());
  if (!isa<mlir::rise::TransposeOp>(apply2.getOperand(0).getDefiningOp())) return Failure();
  auto transpose2 = cast<mlir::rise::TransposeOp>(apply2.getOperand(0).getDefiningOp());
  if (!apply2.getOperand(1).isa<OpResult>()) return Failure();
  // successful match

  Value result = apply2.getOperand(1);

  return success(result.getDefiningOp());
}
auto mlir::elevate::removeTransposePair() -> RemoveTransposePairRewritePattern {return RemoveTransposePairRewritePattern(); }


// utils
void mlir::elevate::substitute(LambdaOp lambda, llvm::SmallVector<Value, 10> args) {
  if (lambda.region().front().getArguments().size() < args.size()) {
    emitError(lambda.getLoc())
        << "Too many arguments given for Lambda substitution";
  }
  for (int i = 0; i < args.size(); i++) {
    lambda.region().front().getArgument(i).replaceAllUsesWith(args[i]);
  }
  return;
}

/* Inline the operations of a Lambda after op */
mlir::Value mlir::elevate::inlineLambda(LambdaOp lambda, mlir::Block *insertionBlock, mlir::Operation *op) {
  mlir::Value lambdaResult =
      lambda.getRegion().front().getTerminator()->getOperand(0);
  insertionBlock->getOperations().splice(
      mlir::Block::iterator(op), lambda.getRegion().front().getOperations(),
      lambda.getRegion().front().begin(),
      mlir::Block::iterator(lambda.getRegion().front().getTerminator()));
  return lambdaResult;
}