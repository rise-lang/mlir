//
// Created by martin on 31/10/2020.
//

#ifndef LLVM_ELEVATE_PREDICATES_H
#define LLVM_ELEVATE_PREDICATES_H

#include "mlir/Dialect/Rise/EDSC/Builders.h"
#include "mlir/Dialect/Rise/EDSC/HighLevel.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Elevate2/core.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace elevate {

class ContainsRewritePattern : public StrategyRewritePattern {
  const Value &val;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
  ContainsRewritePattern(const Value &val) : val(val) {};
};
auto contains(const Value &val) -> ContainsRewritePattern;

class UsesValueRewritePattern : public StrategyRewritePattern {
  const Value &val;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
  UsesValueRewritePattern(const Value &val) : val(val) {};
};
auto usesValue(const Value &val) -> UsesValueRewritePattern;

class EtaReducibleRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
  EtaReducibleRewritePattern() {};
};
auto etaReducible() -> EtaReducibleRewritePattern;

template <typename T>
class IsaRewritePattern : public StrategyRewritePattern {
RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;

public:
IsaRewritePattern<T>() {};
};
template <typename T>
auto _isa() -> IsaRewritePattern<T>;

} // namespace elevate
} // namespace mlir

#endif // LLVM_ELEVATE_PREDICATES_H
