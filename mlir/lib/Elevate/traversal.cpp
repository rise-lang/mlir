//
// Created by martin on 29/10/2020.
//
#include "mlir/Dialect/Rise/Elevate2/traversal.h"

using namespace mlir::rise;
using namespace mlir::elevate;

auto mlir::elevate::argument(const int n, const StrategyRewritePattern &s) -> ArgumentRewritePattern {
  return ArgumentRewritePattern(n, s);
}
auto mlir::elevate::function(const StrategyRewritePattern &s) -> FunctionRewritePattern {
  return FunctionRewritePattern(s);
}
auto mlir::elevate::body(const StrategyRewritePattern &s) -> BodyRewritePattern {
  return BodyRewritePattern(s);
}
auto mlir::elevate::fmap(const StrategyRewritePattern &s) -> FMapRewritePattern {
  return FMapRewritePattern(s);
}
auto mlir::elevate::one(const StrategyRewritePattern &s) -> OneRewritePattern {
  return OneRewritePattern(s);
}
auto mlir::elevate::topdown(const StrategyRewritePattern &s) -> TopDownRewritePattern {
  return TopDownRewritePattern(s);
}
auto mlir::elevate::bottomUp(const StrategyRewritePattern &s) -> BottomUpRewritePattern {
  return BottomUpRewritePattern(s);
}
auto mlir::elevate::normalize(const StrategyRewritePattern &s) -> NormalizeRewritePattern {
  return NormalizeRewritePattern(s);
}
auto mlir::elevate::outermost(const StrategyRewritePattern &predicate, const StrategyRewritePattern &s) -> OutermostRewritePattern {
  return OutermostRewritePattern(predicate, s);
}
auto mlir::elevate::innermost(const StrategyRewritePattern &predicate, const StrategyRewritePattern &s) -> InnermostRewritePattern {
  return InnermostRewritePattern(predicate, s);
}

RewriteResult ArgumentRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op))
    return failure();

  auto apply = cast<ApplyOp>(op);
  auto fun = apply.getOperand(n).getDefiningOp();
  auto rr = s(fun, rewriter);
  if (std::get_if<Failure>(&rr)) return rr;
  return elevate::success(op); 
}

RewriteResult FunctionRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op))
    return failure();

  auto apply = cast<ApplyOp>(op);
  auto fun = apply.getOperand(0).getDefiningOp();
  auto rr = s(fun, rewriter);
  if (std::get_if<Failure>(&rr)) return rr;
  return elevate::success(op);
}

RewriteResult BodyRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<LambdaOp>(op) && !isa<EmbedOp>(op))
    return failure();

  mlir::rise::ReturnOp terminator = dyn_cast<mlir::rise::ReturnOp>(
      op->getRegion(0).front().getTerminator());
  auto rr = s(terminator.getOperand(0).getDefiningOp(), rewriter);
  if (std::get_if<Failure>(&rr)) return rr;
  return success(op);
}

RewriteResult mlir::elevate::FMapRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op))
    return elevate::failure();

  auto apply = cast<ApplyOp>(op);
  if (!isa<MapSeqOp>(apply.getOperand(0).getDefiningOp())) return elevate::failure();

  auto rr = argument(1, body(s))(op, rewriter);
  if (std::get_if<Failure>(&rr)) return rr;
  return success(op);
}

RewriteResult OneRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op) && !isa<LambdaOp>(op) && !isa<EmbedOp>(op))
    return failure();

  if (ApplyOp apply = dyn_cast<ApplyOp>(op)) {
    // try applying s to the operands right to left
    for (int i = apply.getNumOperands() - 1; i >= 0; i--) {
      if (!apply.getOperand(i).isa<OpResult>())
        continue;
      auto operand = apply.getOperand(i).getDefiningOp();
      auto rr = s(operand, rewriter);
      if (std::get_if<Failure>(&rr))
        continue;
      return success(op);
    }
  }
  if (LambdaOp lambda = dyn_cast<LambdaOp>(op)) {
    auto rr = BodyRewritePattern(s)(lambda, rewriter);
    if (std::get_if<Failure>(&rr)) return rr;
    return success(op);
  }
  if (EmbedOp embed = dyn_cast<EmbedOp>(op)) {
    RewriteResult rr = BodyRewritePattern(s)(embed, rewriter);
    if (std::get_if<Success>(&rr))
      return success(op);
    for (int i = embed.getNumOperands() - 1; i >= 0; i--) {
      if (!embed.getOperand(i).isa<OpResult>())
        continue;
      auto operand = embed.getOperand(i).getDefiningOp();
      auto rr = s(operand, rewriter);
      if (std::get_if<Failure>(&rr))
        continue;
      return success(op);
    }
  }
  return Failure();
}

RewriteResult TopDownRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return LeftChoiceRewritePattern(s, OneRewritePattern(TopDownRewritePattern(s)))(op, rewriter);
}

RewriteResult BottomUpRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return LeftChoiceRewritePattern(OneRewritePattern(BottomUpRewritePattern(s)), s)(op, rewriter);
}

RewriteResult NormalizeRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return repeat(topdown(s))(op, rewriter);
}

RewriteResult OutermostRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return topdown(seq(predicate, s))(op, rewriter);
}

RewriteResult InnermostRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return bottomUp(seq(predicate, s))(op, rewriter);
}
