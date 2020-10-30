//
// Created by martin on 9/25/20.
//

#ifndef LLVM_ELEVATE_CORE_H
#define LLVM_ELEVATE_CORE_H

#include <iostream>
#include <memory>
#include <variant>
#include <stdexcept>
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Elevate/ElevateRewriter.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace elevate {

using Expr = mlir::Operation;

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

template <class... Ts>
struct cases : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
cases(Ts...) -> cases<Ts...>;

template <class Variant, class Visitor>
decltype(auto) match(Variant &&var, Visitor &&vis) {
  return std::visit(std::forward<Visitor>(vis), std::forward<Variant>(var));
}

template <class Variant1, class Variant2, class Visitor>
decltype(auto) match(Variant1 &&var1, Variant2 &&var2, Visitor &&vis) {
  return std::visit(std::forward<Visitor>(vis), std::forward<Variant1>(var1),
                    std::forward<Variant2>(var2));
}

struct Success;
struct Failure;

using RewriteResult = std::variant<Success, Failure>;

struct Success {
  Expr &expr;
};

auto success(Expr &expr) -> Success { return Success{expr}; }

struct Failure {};

auto failure() -> Failure { return Failure{}; }

auto getExpr(RewriteResult rr) -> Expr & {
  return match(rr, cases{[](const Success &s) -> Expr & {
    return const_cast<Expr &>(s.expr);
  },
                         [](const Failure &f) -> Expr & {
                           std::cout << "elevate logic error!\n";
                           // throw std::logic_error("");
                           // return {}; // what Expr do we want to return here?
                           // this obviously segfaults
                         }});
}

struct Strategy {
  virtual RewriteResult rewrite(Expr &expr) const = 0;

  RewriteResult operator()(Expr &expr) const {
    RewriteResult rr = rewrite(expr);

    if (auto _ = std::get_if<Failure>(&rr)) return rr;
    Expr &newExpr = getExpr(rr);

    // Check if expr has been deleted or replaced already
    if (&expr == nullptr) return rr;
    if (expr.use_empty()) return rr;
    // Did the strategy modify the IR at all
    if (&expr == &newExpr) return rr;

    auto rewriter = ElevateRewriter::getInstance().rewriter;
    std::vector<Operation *> garbageCandidates;
    auto addOperandsToGarbageCandidates = [&](Operation *op) {
      llvm::for_each(op->getOperands(), [&](Value operand) {
        if (auto opResult = operand.dyn_cast<OpResult>()) {
          garbageCandidates.push_back(opResult.getDefiningOp());
        }});
    };

    addOperandsToGarbageCandidates(&expr);
    expr.replaceAllUsesWith(&newExpr);
    rewriter->eraseOp(&expr);
    do {
      auto currentOp = garbageCandidates.back();
      garbageCandidates.pop_back();

      addOperandsToGarbageCandidates(currentOp);
      if (currentOp->use_empty()) {
        rewriter->eraseOp(currentOp);
      }
    } while (!garbageCandidates.empty());

    return success(newExpr);
  }
};

auto flatMapSuccess(RewriteResult rr, const Strategy &s) -> RewriteResult {
  return match(
      rr, cases{[&](const Success &ss) -> RewriteResult { return s(ss.expr); },
                [&](const Failure &f) -> RewriteResult { return f; }});
}

auto mapSuccess(RewriteResult rr, std::function<Expr&(Expr&)> f) -> RewriteResult {
  return match(rr, cases{[&](const Success &ss) -> RewriteResult { return success(f(ss.expr)); },
                         [&](const Failure &f) -> RewriteResult { return f; }});
}

template <typename F>
auto flatMapFailure(RewriteResult rr, const F &f) -> RewriteResult {
  return match(rr,
               cases{[&](const Success &ss) -> RewriteResult { return ss; },
                     [&](const Failure &) -> RewriteResult { return f(); }});
}

struct IdStrategy : Strategy {
  RewriteResult rewrite(Expr &expr) const override { return success(expr); };
};

auto id = IdStrategy();

struct DebugStrategy : Strategy {
  const std::string msg;
  DebugStrategy(const std::string msg) : msg{msg} {};

  RewriteResult rewrite(Expr &expr) const override {
    if (FileLineColLoc loc = expr.getLoc().dyn_cast<FileLineColLoc>()) {
      llvm::dbgs() << loc.getFilename().str() << ":" << loc.getLine() << ":" << loc.getColumn() << " ";
    }
    llvm::dbgs() << expr.getName().getStringRef().str() << ": " << msg.c_str() << "\n";
    return success(expr);
  };
};

auto debug = [](const auto &msg) { return DebugStrategy(msg); };
auto debug2 = DebugStrategy("");

struct FailStrategy : Strategy {
  RewriteResult rewrite(Expr &expr) const override { return failure(); };
};

auto fail = FailStrategy();

struct SeqStrategy : Strategy {
  const Strategy &fs;
  const Strategy &ss;

  SeqStrategy(const Strategy &fs, const Strategy &ss) : fs{fs}, ss{ss} {};

  RewriteResult rewrite(Expr &expr) const override {
    return flatMapSuccess(fs(expr), ss);
  };
};

auto seq = [](const auto &fs) {
  return [&](const auto &ss) { return SeqStrategy(fs, ss); };
};

struct LeftChoiceStrategy : Strategy {
  const Strategy &fs;
  const Strategy &ss;

  LeftChoiceStrategy(const Strategy &fs, const Strategy &ss)
      : fs{fs}, ss{ss} {};

  RewriteResult rewrite(Expr &expr) const override {
    return flatMapFailure(fs(expr), [&] { return ss(expr); });
  };
};

auto leftChoice = [](const auto &fs) {
  return [&](const auto &ss) { return LeftChoiceStrategy(fs, ss); };
};

struct TryStrategy : Strategy {
  const Strategy &s;

  TryStrategy(const Strategy &s) : s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    return leftChoice(s)(id)(expr);
  };
};

auto try_ = [](const auto &s) { return TryStrategy(s); };

struct RepeatStrategy : Strategy {
  const Strategy &s;

  RepeatStrategy(const Strategy &s) : s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    return try_(seq(s)(RepeatStrategy(s)))(expr);
  };
};

auto repeat = [](const auto &s) { return RepeatStrategy(s); };

}
}

#endif // LLVM_ELEVATE_CORE_H
