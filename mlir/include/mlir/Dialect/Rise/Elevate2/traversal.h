//
// Created by martin on 29/10/2020.
//

#ifndef LLVM_TRAVERSAL_H
#define LLVM_TRAVERSAL_H

#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Elevate2/core.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"

namespace mlir {
namespace elevate {

class ArgumentRewritePattern : public StrategyRewritePattern {
  const int n;
  const StrategyRewritePattern &s;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
  ArgumentRewritePattern(const int n, const StrategyRewritePattern &s)
      : s(s), n(n) {}
};
auto argument(const int n, const StrategyRewritePattern &s)
    -> ArgumentRewritePattern;

class FunctionRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &s;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
  FunctionRewritePattern(const StrategyRewritePattern &s) : s(s) {}
};
auto function(const StrategyRewritePattern &s) -> FunctionRewritePattern;

class BodyRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &s;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
  BodyRewritePattern(const StrategyRewritePattern &s) : s(s) {}
};
auto body(const StrategyRewritePattern &s) -> BodyRewritePattern;

class FMapRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &s;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
  FMapRewritePattern(const StrategyRewritePattern &s) : s(s) {}
};
auto fmap(const StrategyRewritePattern &s) -> FMapRewritePattern;

class OneRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &s;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
  OneRewritePattern(const StrategyRewritePattern &s) : s(s) {}
};
auto one(const StrategyRewritePattern &s) -> OneRewritePattern;

class TopDownRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &s;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
  TopDownRewritePattern(const StrategyRewritePattern &s) : s(s) {}
};
auto topdown(const StrategyRewritePattern &s) -> TopDownRewritePattern;

class BottomUpRewritePattern : public StrategyRewritePattern {
const StrategyRewritePattern &s;

RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
BottomUpRewritePattern(const StrategyRewritePattern &s) : s(s) {}
};
auto bottomUp(const StrategyRewritePattern &s) -> BottomUpRewritePattern;

class NormalizeRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &s;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
  NormalizeRewritePattern(const StrategyRewritePattern &s) : s(s) {}
};
auto normalize(const StrategyRewritePattern &s) -> NormalizeRewritePattern;

class OutermostRewritePattern : public StrategyRewritePattern {
const StrategyRewritePattern &predicate;
const StrategyRewritePattern &s;

RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
OutermostRewritePattern(const StrategyRewritePattern &predicate, const StrategyRewritePattern &s) : predicate(predicate), s(s) {}
};
auto outermost(const StrategyRewritePattern &predicate, const StrategyRewritePattern &s) -> OutermostRewritePattern;

class InnermostRewritePattern : public StrategyRewritePattern {
const StrategyRewritePattern &predicate;
const StrategyRewritePattern &s;

RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
InnermostRewritePattern(const StrategyRewritePattern &predicate, const StrategyRewritePattern &s) : predicate(predicate), s(s) {}
};
auto innermost(const StrategyRewritePattern &predicate, const StrategyRewritePattern &s) -> InnermostRewritePattern;



} // namespace elevate
} // namespace mlir

#endif // LLVM_TRAVERSAL_H
