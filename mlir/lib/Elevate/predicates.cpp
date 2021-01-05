//
// Created by martin on 31/10/2020.
//

#include "mlir/Dialect/Rise/Elevate2/predicates.h"
#include "mlir/Dialect/Rise/Elevate2/traversal.h"

using namespace mlir::rise;
using namespace mlir::edsc::abstraction;
using namespace mlir::edsc::op;
using namespace mlir::edsc::type;
using namespace mlir::elevate;

RewriteResult mlir::elevate::UsesValueRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  auto sameVal = llvm::find(op->getOperands(), val);
  if (sameVal == op->getOperands().end())
    return Failure();
  return success(op);
}
auto mlir::elevate::usesValue(const Value &val) -> UsesValueRewritePattern {
  return UsesValueRewritePattern(val);
}

RewriteResult mlir::elevate::ContainsRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  auto rr = topdown(usesValue(val))(op, rewriter);
  if (auto _ = std::get_if<Failure>(&rr)) {
    return Failure();
  }
  return success(op);
}
auto mlir::elevate::contains(const Value &val) -> ContainsRewritePattern {
  return ContainsRewritePattern(val);
}

RewriteResult mlir::elevate::EtaReducibleRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<LambdaOp>(op)) return Failure();
  auto outerLambda = cast<LambdaOp>(op);
  if (!isa<ApplyOp>(outerLambda.region().front().getTerminator()->getOperand(0).getDefiningOp()));
  auto apply = outerLambda.region().front().getTerminator()->getOperand(0).getDefiningOp();

  // check that region of lambda does not contain the arg of the lambda
  auto rrContainsArg = contains(outerLambda.region().front().getArgument(0))(op, rewriter);
  if (auto _ =  std::get_if<Success>(&rrContainsArg)) {
    return Failure();
  }
  return success(op);
}
// set out identity lambda and another arg around it.
// and pass that lambda into a map
// BENF should remove 1 of the lambdas
auto mlir::elevate::etaReducible() -> EtaReducibleRewritePattern {
  return EtaReducibleRewritePattern();
}
template <typename T>
RewriteResult mlir::elevate::IsaRewritePattern<T>::impl(
    Operation *op, PatternRewriter &rewriter) const {
  if (mlir::isa<T>(op)) {
    return success(op);
  } else {
    return Failure();
  }
}
template <typename T>
auto mlir::elevate::_isa() -> IsaRewritePattern<T> {
  return IsaRewritePattern<T>();
}