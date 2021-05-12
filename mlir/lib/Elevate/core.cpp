//
// Created by martin on 9/25/20.
//


#include "mlir/Elevate2/core.h"
#include "mlir/Elevate/ElevateRewriter.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <variant>
#include "mlir/Dialect/Rise/IR/Dialect.h"

using namespace mlir::elevate;

#define DEBUG_TYPE "elevate"

mlir::LogicalResult
StrategyRewritePattern::matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
  RewriteResult rr = this->operator()(op, rewriter);
  if (std::get_if<Failure>(&rr))
    return mlir::failure();
  return mlir::success();
}

RewriteResult StrategyRewritePattern::operator()(Operation *op, PatternRewriter &rewriter) const {
  if (!op) {
    LLVM_DEBUG({
      llvm::dbgs() << "Strategy called on invalid op!";
    });
    return Failure();
  }
  RewriteResult rr = impl(op, rewriter);

  if (auto _ = std::get_if<Failure>(&rr)) return rr;

  auto ss = std::get<Success>(rr);
  Operation *newOp = ss.op;

  // Check if expr has been deleted or replaced already
  if (op == nullptr) return rr;
  if (op->getBlock() == nullptr) return rr;
  if (op->use_empty()) return rr;
  // Did the strategy modify the IR at all
  if (op == newOp) return rr;


  std::vector<Operation *> garbageCandidates;
  auto addOperandsToGarbageCandidates = [&](Operation *op) {
    llvm::for_each(op->getOperands(), [&](Value operand) {
      if (auto opResult = operand.dyn_cast<OpResult>()) {
        garbageCandidates.push_back(opResult.getDefiningOp());
      }});
  };

  addOperandsToGarbageCandidates(op);

  op->replaceAllUsesWith(newOp);
  rewriter.eraseOp(op);
  do {
    auto currentOp = garbageCandidates.back();
    garbageCandidates.pop_back();

    addOperandsToGarbageCandidates(currentOp);
    if (currentOp->use_empty()) {
//      llvm::dbgs() << "erasing " << currentOp->getName().getStringRef() << "\n";
      rewriter.eraseOp(currentOp);
    }
  } while (!garbageCandidates.empty());

  return success(newOp);
}

// move these?
auto mlir::elevate::success(Operation *op) -> Success { return Success{op}; }
auto mlir::elevate::failure() -> Failure { return Failure{}; }

auto mlir::elevate::flatMapSuccess(RewriteResult rr, const StrategyRewritePattern &s, PatternRewriter &rewriter) -> RewriteResult {
  return match(
      rr, cases{[&](const Success &ss) -> RewriteResult { return s(ss.op, rewriter); },
                [&](const Failure &f) -> RewriteResult { return f; }});
}

auto mlir::elevate::mapSuccess(RewriteResult rr, PatternRewriter &rewriter, std::function<Operation*(Operation*, PatternRewriter&)> f) -> RewriteResult {
  return match(rr, cases{[&](const Success &ss) -> RewriteResult { return mlir::elevate::success(f(ss.op, rewriter)); },
                         [&](const Failure &f) -> RewriteResult { return f; }});
}

template <typename F>
auto mlir::elevate::flatMapFailure(RewriteResult rr, const F &f) -> RewriteResult {
  return match(rr,
               cases{[&](const Success &ss) -> RewriteResult { return ss; },
                     [&](const Failure &) -> RewriteResult { return f(); }});
}

auto mlir::elevate::id() -> IdRewritePattern { return IdRewritePattern(); }
auto mlir::elevate::fail() -> FailRewritePattern {
  return FailRewritePattern();
}
auto mlir::elevate::seq(const StrategyRewritePattern &fs,
                         const StrategyRewritePattern &ss)
    -> SeqRewritePattern {
  return SeqRewritePattern(fs, ss);
}
auto mlir::elevate::debug(const std::string &msg) -> DebugRewritePattern {
  return DebugRewritePattern(msg);
}
auto mlir::elevate::leftChoice(const StrategyRewritePattern &fs,
                                const StrategyRewritePattern &ss)
    -> LeftChoiceRewritePattern {
  return LeftChoiceRewritePattern(fs, ss);
}
auto mlir::elevate::try_(const StrategyRewritePattern &s) -> TryRewritePattern {
  return TryRewritePattern(s);
}
auto mlir::elevate::repeat(const StrategyRewritePattern &s) -> RepeatRewritePattern {
  return RepeatRewritePattern(s);
}

RewriteResult IdRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return success(op);
}

RewriteResult FailRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return failure();
}

RewriteResult DebugRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (FileLineColLoc loc = op->getLoc().dyn_cast<FileLineColLoc>()) {
    llvm::dbgs() << loc.getFilename().str() << ":" << loc.getLine() << ":" << loc.getColumn() << " ";
  }
  llvm::dbgs() << op->getName().getStringRef().str() << ": " << msg.c_str() << "\n";
  return success(op);
}

RewriteResult SeqRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  // has members   const StrategyRewritePattern &fs;
  // and           const StrategyRewritePattern &ss;
  return flatMapSuccess(fs(op, rewriter), ss, rewriter);
}

RewriteResult LeftChoiceRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return flatMapFailure(fs(op, rewriter), [&] { return ss(op, rewriter); });
}

RewriteResult TryRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return LeftChoiceRewritePattern(s, IdRewritePattern())(op, rewriter);
}

RewriteResult RepeatRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return try_(seq(s, repeat(s)))(op, rewriter);
}


