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
  // check for valid order is usually private. For some reason the op is corrupted and
  // I can't check that properly here
  if (!op->hasValidOrder())
    op->updateOrderIfNecessary();
  if (!op || op->getBlock() == nullptr || !op->hasValidOrder()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Strategy " << getName() << " called on invalid op!\n";
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

  // TODO: some strategy returns not the correct op and we replace some other op
  // check which strat is applied to the wrongly replaced op!

//  llvm::dbgs() << "replacing op for strat " << getName() << " :\n";
  if (op) llvm::dbgs() << "op there\n";

  if (getName().equals("seq") || getName().equals("repeat") || getName().equals("try") || getName().equals("<+") || getName().equals("normalize"))
    return rr;
//  llvm::dbgs() << "op: " << op->getName().getStringRef().str() << "\n";
//  op->dump();
//  debug("opp:", {50,150,30})(op, rewriter);

  op->replaceAllUsesWith(newOp);
  rewriter.eraseOp(op);
  do {
    auto currentOp = garbageCandidates.back();
    garbageCandidates.pop_back();

    addOperandsToGarbageCandidates(currentOp);
    if (currentOp->use_empty()) {
//      llvm::dbgs() << "erasing " << currentOp->getName().getStringRef() << "\n";
//      currentOp->dump();

      currentOp->dropAllUses();
//      if (rise::LambdaOp lambda = dyn_cast<rise::LambdaOp>(currentOp)) {
//        if (!lambda) break;
//        auto lambdaArg = lambda.region().getArgument(0);
//        lambdaArg.dump();
//        for (auto use : lambdaArg.getUsers()) {
//          use->dump();
//          use->getParentOp()->dump();
//        }
//      }
  // TODO:
  // catch weird behaviour:
  // This should not be necessary. I want this to be debuggable easier
//      if (currentOp == nullptr) { continue; }
//      if (currentOp->getBlock() == nullptr) {
//        llvm::dbgs() << "Has no block!\n";
//        continue;
//      }
//      if (currentOp->getParentOp() == nullptr) {
//        llvm::dbgs() << "Has no Parent!\n";
//        continue;
//      }

//      debug("lambda?", {50,150,30})(currentOp, rewriter);

      // one of the blockargs of this lambda (or one nested inside) has a use somewhere
      rewriter.eraseOp(currentOp);
//      llvm::dbgs() << "erasing successful!\n";
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
auto mlir::elevate::debug(const std::string &msg, std::tuple<float,float,float> color) -> DebugRewritePattern {
  return DebugRewritePattern(msg, color);
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
auto mlir::elevate::repeatNTimes(const size_t n, const StrategyRewritePattern &s) -> RepeatNTimesRewritePattern {
  return RepeatNTimesRewritePattern(n, s);
}

RewriteResult IdRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return success(op);
}
llvm::StringRef IdRewritePattern::getName() const {return llvm::StringRef("id");}

RewriteResult FailRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return failure();
}
llvm::StringRef FailRewritePattern::getName() const {return llvm::StringRef("fail");}

RewriteResult DebugRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (FileLineColLoc loc = op->getLoc().dyn_cast<FileLineColLoc>()) {
    llvm::dbgs() << loc.getFilename().str() << ":" << loc.getLine() << ":" << loc.getColumn() << " ";
  }
  llvm::dbgs() << op->getName().getStringRef().str() << ": " << msg.c_str() << "\n";
  op->dump();
  if (rise::ApplyOp apply = dyn_cast<rise::ApplyOp>(op)) {
    apply.fun().dump();
  }
  // if debug printing a rise op, print it in the whole context of the whole rise program
  if (!op->getDialect()->getNamespace().equals(rise::RiseDialect::getDialectNamespace()))
    return success(op);
  rise::LoweringUnitOp loweringUnit = op->getParentOfType<rise::LoweringUnitOp>();
  // Start at the back and find the rise.out op
  Block &block = loweringUnit.region().front();
  auto _outOp = std::find_if(block.rbegin(), block.rend(),
                             [](auto &op) { return isa<rise::OutOp>(op); });
  if (_outOp == block.rend()) {
    emitWarning(loweringUnit.getLoc()) << "Could not find rise.out operation for debug printing!";
    return success(op);
  }
  rise::OutOp outOp = dyn_cast<rise::OutOp>(*_outOp);
  auto lastApply = outOp.input().getDefiningOp();
  rise::RiseDialect::dumpRiseExpression(lastApply,false,op->getResult(0), color);

  return success(op);
}
llvm::StringRef DebugRewritePattern::getName() const {return llvm::StringRef("debug");}

RewriteResult SeqRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  // has members   const StrategyRewritePattern &fs;
  // and           const StrategyRewritePattern &ss;
  return flatMapSuccess(fs(op, rewriter), ss, rewriter);
}
llvm::StringRef SeqRewritePattern::getName() const {return llvm::StringRef("seq");}

RewriteResult LeftChoiceRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return flatMapFailure(fs(op, rewriter), [&] { return ss(op, rewriter); });
}
llvm::StringRef LeftChoiceRewritePattern::getName() const {return llvm::StringRef("<+");}

RewriteResult TryRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return LeftChoiceRewritePattern(s, IdRewritePattern())(op, rewriter);
}
llvm::StringRef TryRewritePattern::getName() const {return llvm::StringRef("try");}

RewriteResult RepeatRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  return try_(seq(s, repeat(s)))(op, rewriter);
}
llvm::StringRef RepeatRewritePattern::getName() const {return llvm::StringRef("repeat");}

RewriteResult RepeatNTimesRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (n == 0)
    return id()(op, rewriter);
  return seq(s, repeatNTimes(n - 1, s))(op, rewriter);
}
llvm::StringRef RepeatNTimesRewritePattern::getName() const {return llvm::StringRef("repeatNTimes");}
