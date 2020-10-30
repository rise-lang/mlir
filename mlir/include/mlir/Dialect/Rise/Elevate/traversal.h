//
// Created by martin on 9/25/20.
//

#ifndef LLVM_ELEVATE_TRAVERSAL_H
#define LLVM_ELEVATE_TRAVERSAL_H

#include "mlir/Elevate/core.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"

using namespace mlir;
using namespace mlir::rise;
using namespace mlir::elevate;

struct BodyStrategy : Strategy {
  const Strategy &s;
  BodyStrategy(const Strategy &s) : s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    if (!isa<LambdaOp>(expr) && !isa<EmbedOp>(expr))
      return elevate::failure();

    mlir::rise::ReturnOp terminator = dyn_cast<mlir::rise::ReturnOp>(
        expr.getRegion(0).front().getTerminator());
    auto rr = s(*terminator.getOperand(0).getDefiningOp());
    if (std::get_if<Failure>(&rr)) return rr;
    return success(expr);
  };
};

auto body = [](const auto &s) { return BodyStrategy(s); };

struct FunctionStrategy : Strategy {
  const Strategy &s;
  FunctionStrategy(const Strategy &s) : s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    if (!isa<ApplyOp>(expr))
      return elevate::failure();

    auto apply = cast<ApplyOp>(expr);
    auto fun = apply.getOperand(0).getDefiningOp();
    auto rr = s(*fun);
    if (std::get_if<Failure>(&rr)) return rr;
    return success(expr);
  };
};

auto function = [](const auto &s) { return FunctionStrategy(s); };

struct ArgumentStrategy : Strategy {
  int n;
  const Strategy &s;
  ArgumentStrategy(const int n, const Strategy &s) : n{n}, s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    if (!isa<ApplyOp>(expr))
      return elevate::failure();

    auto apply = cast<ApplyOp>(expr);
    if (!apply.getOperand(n).isa<OpResult>()) return Failure();
    auto arg = apply.getOperand(n).getDefiningOp();
    auto rr = s(*arg);
    if (std::get_if<Failure>(&rr)) return rr;
    return success(expr);
  };
};

auto argument = [](const auto n, const auto &s) {
  return ArgumentStrategy(n, s);
};

struct FMapStrategy : Strategy {
  const Strategy &s;
  FMapStrategy(const Strategy &s) : s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    llvm::dbgs() << "fmapp!\n";
    if (!isa<ApplyOp>(expr))
      return elevate::failure();

    auto apply = cast<ApplyOp>(expr);
    if (!isa<MapSeqOp>(apply.getOperand(0).getDefiningOp())) return elevate::failure();
    llvm::dbgs() << "fmapp!\n";

    auto rr = argument(1, body(s))(expr);
    if (std::get_if<Failure>(&rr)) return rr;
    return success(expr);
  };
};

auto fmap = [](const auto &s) {
  return FMapStrategy(s);
};

struct OneStrategy : Strategy {
  const Strategy &s;
  OneStrategy(const Strategy &s) : s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    if (!isa<ApplyOp>(expr) && !isa<LambdaOp>(expr) && !isa<EmbedOp>(expr))
      return elevate::failure();

    if (ApplyOp apply = dyn_cast<ApplyOp>(expr)) {
      // try applying s to the operands right to left
      for (int i = apply.getNumOperands() - 1; i >= 0; i--) {
        if (!apply.getOperand(i).isa<OpResult>())
          continue;
        auto op = apply.getOperand(i).getDefiningOp();
        auto rr = s(*op);
        if (std::get_if<Failure>(&rr))
          continue;
        return success(expr);
      }
    }
    if (LambdaOp lambda = dyn_cast<LambdaOp>(expr)) {
      auto rr = body(s)(*lambda);
      if (std::get_if<Failure>(&rr)) return rr;
      return success(expr);
    }
    if (EmbedOp embed = dyn_cast<EmbedOp>(expr)) {
      RewriteResult rr = body(s)(*embed);
      if (std::get_if<Success>(&rr))
        return success(expr);
      for (int i = embed.getNumOperands() - 1; i >= 0; i--) {
        if (!embed.getOperand(i).isa<OpResult>())
          continue;
        auto op = embed.getOperand(i).getDefiningOp();
        auto rr = s(*op);
        if (std::get_if<Failure>(&rr))
          continue;
        return success(expr);
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

  RewriteResult rewrite(Expr &expr) const override {
    return leftChoice(s)(one(TopDownStrategy(s)))(expr);
  };
};

auto topdown = [](const Strategy &s) { return TopDownStrategy(s); };

struct BottomUpStrategy : Strategy {
  const Strategy &s;
  BottomUpStrategy(const Strategy &s) : s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    return leftChoice(one(BottomUpStrategy(s)))(s)(expr);
  };
};

auto bottomUp = [](const Strategy &s) { return BottomUpStrategy(s); };

// package
struct NormalizeStrategy : Strategy {
  const Strategy &s;

  NormalizeStrategy(const Strategy &s) : s{s} {};

  RewriteResult rewrite(Expr &expr) const override {
    return repeat(topdown(s))(expr);
  };
};

auto normalize = [](const auto &s) { return NormalizeStrategy(s); };

// in place of isEqualTo strategy
struct usesValueStrategy : Strategy {
  const Value &val;

  usesValueStrategy(const Value &val) : val{val} {};

  RewriteResult rewrite(Expr &expr) const override {
      auto sameVal = llvm::find(expr.getOperands(), val);
      if (sameVal == expr.getOperands().end()) return Failure();
      return success(expr);
  };
};

auto usesValue = [](const auto &val) { return usesValueStrategy(val); };

struct ContainsStrategy : Strategy {
  const Value &val;

  ContainsStrategy(const Value &val) : val{val} {};

  RewriteResult rewrite(Expr &expr) const override {
    return topdown(usesValue(val))(expr);
  };
};

auto contains = [](const auto &val) { return ContainsStrategy(val); };


// utils
void substitute(LambdaOp lambda, llvm::SmallVector<Value, 10> args) {
  if (lambda.region().front().getArguments().size() < args.size()) {
    emitError(lambda.getLoc())
        << "Too many arguments given for Lambda substitution";
  }
  for (int i = 0; i < args.size(); i++) {
    lambda.region().front().getArgument(i).replaceAllUsesWith(args[i]);
  }
  return;
}

/* Inline the operations of a Lambda after op */
Value inlineLambda(LambdaOp lambda, Block *insertionBlock, Operation *op) {
  Value lambdaResult = lambda.getRegion().front().getTerminator()->getOperand(0);
  insertionBlock->getOperations().splice(
      Block::iterator(op),
      lambda.getRegion().front().getOperations(),
      lambda.getRegion().front().begin(), Block::iterator(lambda.getRegion().front().getTerminator()));
  return lambdaResult;
}

struct BetaReductionStrategy : Strategy {
  BetaReductionStrategy(){};

  RewriteResult rewrite(Expr &expr) const override {
    if (!isa<ApplyOp>(expr)) return Failure();
    auto apply = cast<ApplyOp>(expr);
    if (!isa<LambdaOp>(apply.getOperand(0).getDefiningOp())) return Failure();
    // match success
    PatternRewriter &rewriter = *ElevateRewriter::getInstance().rewriter;
    auto lambda = cast<LambdaOp>(apply.getOperand(0).getDefiningOp());

    SmallVector<Value, 10> args = SmallVector<Value, 10>();
    for (int i = 1; i < apply.getNumOperands(); i++) {
        args.push_back(apply.getOperand(i));
    }
    substitute(lambda, args);
    Value inlinedLambdaResult = inlineLambda(lambda, expr.getBlock(), apply);
    Expr *newExpr = inlinedLambdaResult.getDefiningOp();

    return success(*newExpr);
  };
};

auto betaReduction = BetaReductionStrategy();

struct EtaReducibleStrategy : Strategy {
  EtaReducibleStrategy(){};

  RewriteResult rewrite(Expr &expr) const override {
    if (!isa<LambdaOp>(expr)) return Failure();
    auto outerLambda = cast<LambdaOp>(expr);
    if (!isa<ApplyOp>(outerLambda.region().front().getTerminator()->getOperand(0).getDefiningOp()));
    auto apply = outerLambda.region().front().getTerminator()->getOperand(0).getDefiningOp();

    // check that region of lambda does not contain the arg of the lambda
    auto rrContainsArg = contains(outerLambda.region().front().getArgument(0))(expr);
    if (auto _ =  std::get_if<Success>(&rrContainsArg)) {
      return Failure();
    }
    return success(expr);
  };
};

auto etaReducible = EtaReducibleStrategy();

#endif // LLVM_ELEVATE_TRAVERSAL_H
