//
// Created by martin on 9/25/20.
//

#ifndef LLVM_ELEVATE_TRAVERSAL_H
#define LLVM_ELEVATE_TRAVERSAL_H

#include "core.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::rise;
using namespace mlir::elevate;

struct BodyStrategy : Strategy {
  const Strategy &s;
  BodyStrategy(const Strategy &s) : s{s} {};

  RewriteResult operator()(Expr &expr) const override {
    if (!isa<LambdaOp>(expr) && !isa<EmbedOp>(expr))
      return elevate::failure();

    mlir::rise::ReturnOp terminator = dyn_cast<mlir::rise::ReturnOp>(
        expr.getRegion(0).front().getTerminator());
    return s(*terminator.getOperand(0).getDefiningOp());
  };
};

auto body = [](const auto &s) { return BodyStrategy(s); };

struct FunctionStrategy : Strategy {
  const Strategy &s;
  FunctionStrategy(const Strategy &s) : s{s} {};

  RewriteResult operator()(Expr &expr) const override {
    if (!isa<ApplyOp>(expr))
      return elevate::failure();

    auto apply = cast<ApplyOp>(expr);
    auto fun = apply.getOperand(0).getDefiningOp();
    return s(*fun);
  };
};

auto function = [](const auto &s) { return FunctionStrategy(s); };

struct ArgumentStrategy : Strategy {
  int n;
  const Strategy &s;
  ArgumentStrategy(const int n, const Strategy &s) : n{n}, s{s} {};

  RewriteResult operator()(Expr &expr) const override {
    if (!isa<ApplyOp>(expr))
      return elevate::failure();

    auto apply = cast<ApplyOp>(expr);
    if (!apply.getOperand(n).isa<OpResult>()) return Failure();
    auto arg = apply.getOperand(n).getDefiningOp();
    return s(*arg);
  };
};

auto argument = [](const auto n, const auto &s) {
  return ArgumentStrategy(n, s);
};

struct OneStrategy : Strategy {
  const Strategy &s;
  OneStrategy(const Strategy &s) : s{s} {};

  RewriteResult operator()(Expr &expr) const override {
    if (!isa<ApplyOp>(expr) && !isa<LambdaOp>(expr) && !isa<EmbedOp>(expr))
      return elevate::failure();

    if (ApplyOp apply = dyn_cast<ApplyOp>(expr)) {
      // try applying s to the operands right to left
      for (int i = apply.getNumOperands() - 1; i >= 0; i--) {
        if (!apply.getOperand(i).isa<OpResult>())
          continue;
        auto op = apply.getOperand(i).getDefiningOp();
        RewriteResult rr = s(*op);
        if (auto success = std::get_if<Success>(&rr)) {
          // Does this have to happen here or in the strategy
          apply.setOperand(i, getExpr(*success).getResult(0));
          return rr;
        }
      }
    }
    if (LambdaOp lambda = dyn_cast<LambdaOp>(expr)) {
      return body(s)(*lambda);
    }
    if (EmbedOp embed = dyn_cast<EmbedOp>(expr)) {
      RewriteResult rr = body(s)(*embed);
      if (std::get_if<Success>(&rr))
        return rr;
      for (int i = embed.getNumOperands() - 1; i >= 0; i--) {
        if (!embed.getOperand(i).isa<OpResult>())
          continue;
        auto op = embed.getOperand(i).getDefiningOp();
        RewriteResult rr = s(*op);
        if (auto success = std::get_if<Success>(&rr)) {
          embed.setOperand(i, getExpr(*success).getResult(0));
          return rr;
        }
      }
    }
    return Failure();
  };
};

auto one = [](const auto &s) { return OneStrategy(s); };

// not rise specific

struct TopDownStrategy : Strategy {
  const Strategy &s;
  TopDownStrategy(const Strategy &s) : s{s} {};

  RewriteResult operator()(Expr &expr) const override {
    return leftChoice(s)(one(TopDownStrategy(s)))(expr);
  };
};

auto topdown = [](const Strategy &s) { return TopDownStrategy(s); };

struct BottomUpStrategy : Strategy {
  const Strategy &s;
  BottomUpStrategy(const Strategy &s) : s{s} {};

  RewriteResult operator()(Expr &expr) const override {
    return leftChoice(one(BottomUpStrategy(s)))(s)(expr);
  };
};

auto bottomUp = [](const Strategy &s) { return BottomUpStrategy(s); };
#endif // LLVM_ELEVATE_TRAVERSAL_H
