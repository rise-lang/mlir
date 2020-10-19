//
// Created by martin on 9/25/20.
//

#ifndef LLVM_ELEVATE_CORE_H
#define LLVM_ELEVATE_CORE_H

#include <iostream>
#include <memory>
#include <variant>
#include <stdexcept>
#include "mlir/IR/OpDefinition.h"
#include "mlir/Elevate/ElevateRewriter.h"

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
  virtual bool traversal() const = 0;

  RewriteResult operator()(Expr &expr) const {
    RewriteResult rr = rewrite(expr);
    if (traversal()) return rr;
    if (auto _ = std::get_if<Failure>(&rr)) return rr;

    Expr &newExpr = getExpr(rr);
    auto rewriter = ElevateRewriter::getInstance().rewriter;
    // clean up here:
    expr.replaceAllUsesWith(&newExpr);

    // collect garbage and dispose
    std::vector<Operation *> worklist;
    auto addOperandsToWorklist = [&](Operation *op) {
      llvm::for_each(op->getOperands(), [&](Value operand) {
        if (auto opResult = operand.dyn_cast<OpResult>()) {
//          std::cout << "pushing back:" << operand.getDefiningOp()->getName().getStringRef().str() << "\n" << std::flush;
          worklist.push_back(opResult.getDefiningOp());
        }});
    };
    addOperandsToWorklist(&expr);

    rewriter->eraseOp(&expr);
    do {
      auto currentOp = worklist.back();
      worklist.pop_back();

      addOperandsToWorklist(currentOp);
      if (currentOp->use_empty()) {
//        std::cout << "erasing op: " << currentOp->getName().getStringRef().str() << "\n" << std::flush;
        rewriter->eraseOp(currentOp);
      }
    } while (!worklist.empty());

    return success(newExpr);
  }
};

// TODO:
// automatic cleanup will be possible. We will always replace the matched operation with what ever the strategy returns.
// We then check for all previous operands, if they have still uses and remove them accordingly



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
  bool traversal() const override {return true;};
  RewriteResult rewrite(Expr &expr) const override { return success(expr); };
};

auto id = IdStrategy();

struct DebugStrategy : Strategy {
  const std::string msg;
  bool traversal() const override {return true;};

  DebugStrategy(const std::string msg) : msg{msg} {};

  RewriteResult rewrite(Expr &expr) const override {
    if (FileLineColLoc loc = expr.getLoc().dyn_cast<FileLineColLoc>()) {
      std::cout << loc.getFilename().str() << ":" << loc.getLine() << ":" << loc.getColumn() << " ";
    }
    std::cout << expr.getName().getStringRef().str() << ": " << msg.c_str() << "\n";
    return success(expr);
  };
};

auto debug = [](const auto &msg) { return DebugStrategy(msg); };
auto debug2 = DebugStrategy("");

struct FailStrategy : Strategy {
  bool traversal() const override {return true;};
  RewriteResult rewrite(Expr &expr) const override { return failure(); };
};

auto fail = FailStrategy();

struct SeqStrategy : Strategy {
  const Strategy &fs;
  const Strategy &ss;
  bool traversal() const override {return true;};

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
  bool traversal() const override {return true;};

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
  bool traversal() const override {return true;};

  TryStrategy(const Strategy &s) : s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    return leftChoice(s)(id)(expr);
  };
};

auto try_ = [](const auto &s) { return TryStrategy(s); };

struct RepeatStrategy : Strategy {
  const Strategy &s;
  bool traversal() const override {return true;};


  RepeatStrategy(const Strategy &s) : s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    return try_(seq(s)(RepeatStrategy(s)))(expr);
  };
};

auto repeat = [](const auto &s) { return RepeatStrategy(s); };

}
}

#endif // LLVM_ELEVATE_CORE_H
