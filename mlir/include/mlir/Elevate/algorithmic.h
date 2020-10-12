//
// Created by martin on 9/25/20.
//

#ifndef LLVM_ELEVATE_ALGORITHMIC_H
#define LLVM_ELEVATE_ALGORITHMIC_H

#include "core.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/EDSC/Builders.h"
#include "mlir/Dialect/Rise/EDSC/HighLevel.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::rise;
using namespace mlir::edsc::abstraction;
using namespace mlir::edsc::op;
using namespace mlir::edsc::type;
using namespace mlir::elevate;

struct FuseReduceMapStrategy : Strategy {
  FuseReduceMapStrategy() {};

  RewriteResult operator()(Expr &expr) const override {
    PatternRewriter *rewriter = ElevateRewriter::getInstance().rewriter;

    if (!isa<ApplyOp>(expr)) return Failure();
    auto applyReduction = cast<ApplyOp>(expr);

    if (!isa<ReduceSeqOp>(applyReduction.getOperand(0).getDefiningOp())) return Failure();
    auto reduction = cast<ReduceSeqOp>(applyReduction.getOperand(0).getDefiningOp());
    auto reductionLambda = applyReduction.getOperand(1).getDefiningOp();
    auto initializer = applyReduction.getOperand(2).getDefiningOp();

    if (!isa<ApplyOp>(applyReduction.getOperand(3).getDefiningOp())) return Failure();
    auto reductionInput = cast<ApplyOp>(applyReduction.getOperand(3).getDefiningOp());

    if (!isa<MapSeqOp>(reductionInput.getOperand(0).getDefiningOp())) return Failure();
    // successful match
    auto mapSeq = cast<MapSeqOp>(reductionInput.getOperand(0).getDefiningOp());
    auto mapLambda = reductionInput.getOperand(1).getDefiningOp();
    Value mapInput = reductionInput.getOperand(2);

    ScopedContext scope(*rewriter, expr.getLoc());

    Value newReduceApplication = reduceSeq(scalarF32Type(), [&](Value y, Value acc){
          Value mapped = apply(scalarF32Type(), mapLambda->getResult(0), y);
          return apply(scalarF32Type(), reductionLambda->getResult(0), {mapped, acc});
    },initializer->getResult(0), mapInput);

    // cleanup
    expr.replaceAllUsesWith(newReduceApplication.getDefiningOp()); // TODO: factor out
    rewriter->eraseOp(&expr);
    rewriter->eraseOp(reduction);
    rewriter->eraseOp(reductionInput);
    rewriter->eraseOp(mapSeq);

    Operation *result = newReduceApplication.getDefiningOp();

    return success(*result);
  };
};

auto fuseReduceMap = FuseReduceMapStrategy();

#endif // LLVM_ELEVATE_ALGORITHMIC_H
