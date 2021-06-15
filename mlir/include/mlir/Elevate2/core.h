//
// Created by martin on 9/25/20.
//

#ifndef LLVM_ELEVATE_CORE_NEW_H
#define LLVM_ELEVATE_CORE_NEW_H

#include "mlir/Elevate2/RewriteResult.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/ElevateRewriteDriver.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <variant>

namespace mlir {
namespace elevate {

class StrategyRewritePattern : public RewritePattern {
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
  virtual RewriteResult impl(Operation *op, PatternRewriter &rewriter) const = 0;
public:
  virtual llvm::StringRef getName() const = 0;
  RewriteResult operator()(Operation *op, PatternRewriter &rewriter) const;
protected:
  StrategyRewritePattern() : RewritePattern(1, MatchAnyOpTypeTag()) {}
};

auto flatMapSuccess(RewriteResult rr, const StrategyRewritePattern &s, PatternRewriter &rewriter) -> RewriteResult;
auto mapSuccess(RewriteResult rr, PatternRewriter &rewriter, std::function<Operation*(Operation*, PatternRewriter&)> f) -> RewriteResult;
template <typename F>
auto flatMapFailure(RewriteResult rr, const F &f) -> RewriteResult;

class IdRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  IdRewritePattern() {}
};
auto id() -> IdRewritePattern;

class FailRewritePattern : public StrategyRewritePattern {
  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  FailRewritePattern() {}
};
auto fail() -> FailRewritePattern;

class SeqRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &fs;
  const StrategyRewritePattern &ss;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  SeqRewritePattern(const StrategyRewritePattern &fs, const StrategyRewritePattern &ss) : fs(fs), ss(ss) {}
};
auto seq(const StrategyRewritePattern &fs, const StrategyRewritePattern &ss)
    -> SeqRewritePattern;

class DebugRewritePattern : public StrategyRewritePattern {
  const std::string msg;
  std::tuple<float,float,float> color;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  DebugRewritePattern(const std::string msg, std::tuple<float,float,float> color) : msg(msg), color(color) {}
};
auto debug(const std::string &msg, std::tuple<float,float,float> color = {0, 255, 0}) -> DebugRewritePattern;

class LeftChoiceRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &fs;
  const StrategyRewritePattern &ss;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  LeftChoiceRewritePattern(const StrategyRewritePattern &fs, const StrategyRewritePattern &ss)
      : fs(fs), ss(ss) {}
};
auto leftChoice(const StrategyRewritePattern &fs, const StrategyRewritePattern &ss) -> LeftChoiceRewritePattern;

class TryRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &s;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  TryRewritePattern(const StrategyRewritePattern &s) : s(s) {}
};
auto try_(const StrategyRewritePattern &s) -> TryRewritePattern;

class RepeatRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &s;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  RepeatRewritePattern(const StrategyRewritePattern &s) : s(s) {}
};
auto repeat(const StrategyRewritePattern &s) -> RepeatRewritePattern;

class RepeatNTimesRewritePattern : public StrategyRewritePattern {
  const StrategyRewritePattern &s;
  const size_t n;

  RewriteResult impl(Operation *op, PatternRewriter &rewriter) const;
  llvm::StringRef getName() const;
public:
  RepeatNTimesRewritePattern(size_t n, const StrategyRewritePattern &s) : n(n), s(s) {}
};
auto repeatNTimes(const size_t n, const StrategyRewritePattern &s) -> RepeatNTimesRewritePattern;


}
}

#endif // LLVM_ELEVATE_CORE_NEW_H
