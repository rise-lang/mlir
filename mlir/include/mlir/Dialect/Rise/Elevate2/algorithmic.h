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
#include "llvm/Support/Debug.h"

namespace mlir {
namespace elevate {

class SplitJoinRewritePattern : public StrategyRewritePattern {
  int n;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  SplitJoinRewritePattern(const int n) : n(n) {};
};
auto splitJoin(const int n) -> SplitJoinRewritePattern;

class FuseReduceMapRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  FuseReduceMapRewritePattern() {};
};
auto fuseReduceMap() -> FuseReduceMapRewritePattern;

class BetaReductionRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  BetaReductionRewritePattern() {};
};
auto betaReduction() -> BetaReductionRewritePattern;

class AddIdAfterRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  AddIdAfterRewritePattern() {};
};
auto addIdAfter() -> AddIdAfterRewritePattern;

class CreateTransposePairRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  CreateTransposePairRewritePattern() {};
};
auto createTransposePair() -> CreateTransposePairRewritePattern;

class RemoveTransposePairRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  RemoveTransposePairRewritePattern() {};
};
auto removeTransposePair() -> RemoveTransposePairRewritePattern;

class MoveMapMapFBeforeTransposeRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  MoveMapMapFBeforeTransposeRewritePattern() {};
};
auto moveMapMapFBeforeTranspose() -> MoveMapMapFBeforeTransposeRewritePattern;

// fission of the last function to be applied inside a map
// *(g >> .. >> f) -> *(g >> ..) >> *f
class MapLastFissionRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  MapLastFissionRewritePattern() {};
};
auto mapLastFission() -> MapLastFissionRewritePattern;



// utils
void substitute(LambdaOp lambda, llvm::SmallVector<Value, 10> args);
Value inlineLambda(LambdaOp lambda, Block *insertionBlock, Operation *op);
void adjustLambdaType(LambdaOp lambda, PatternRewriter &rewriter);
}
} // namespace mlir

#endif // LLVM_ALGORITHMIC_H
