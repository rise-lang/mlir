//
// Created by martin on 30/10/2020.
//

#ifndef LLVM_ALGORITHMIC_H
#define LLVM_ALGORITHMIC_H

#include "mlir/Dialect/Rise/EDSC/Builders.h"
#include "mlir/Dialect/Rise/EDSC/HighLevel.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Elevate2/core.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace elevate {

class SplitJoinRewritePattern : public StrategyRewritePattern {
  int n;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
public:
  SplitJoinRewritePattern(const int n) : n(n) {};
};
auto splitJoin(const int n) -> SplitJoinRewritePattern;

class FuseReduceMapRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
public:
  FuseReduceMapRewritePattern() {};
};
auto fuseReduceMap() -> FuseReduceMapRewritePattern;

class BetaReductionRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
public:
  BetaReductionRewritePattern() {};
};
auto betaReduction() -> BetaReductionRewritePattern;

class AddIdAfterRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
public:
  AddIdAfterRewritePattern() {};
};
auto addIdAfter() -> AddIdAfterRewritePattern;

class CreateTransposePairRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
public:
  CreateTransposePairRewritePattern() {};
};
auto createTransposePair() -> CreateTransposePairRewritePattern;

class RemoveTransposePairRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
public:
  RemoveTransposePairRewritePattern() {};
};
auto removeTransposePair() -> RemoveTransposePairRewritePattern;




// utils
void substitute(LambdaOp lambda, llvm::SmallVector<Value, 10> args);
Value inlineLambda(LambdaOp lambda, Block *insertionBlock, Operation *op);

}
} // namespace mlir

#endif // LLVM_ALGORITHMIC_H
