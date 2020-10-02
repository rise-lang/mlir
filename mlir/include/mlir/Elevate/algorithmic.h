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


//case e @ App(
//    App(App(ReduceX(), op), init), // reduce
//App(App(map(), f), mapArg)     // map
//) =
struct FuseReduceMapStrategy : Strategy {
  FuseReduceMapStrategy() {};

  RewriteResult operator()(Expr &expr) const override {
//            %map10TuplesToInts = rise.mapSeq #rise.nat<4> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
    //        %multipliedArray = rise.apply %map10TuplesToInts, %tupleMulFun, %zippedArrays
    //
    //        //Reduction
    //        %reductionAdd = rise.lambda (%summand0 : !rise.scalar<f32>, %summand1 : !rise.scalar<f32>) -> !rise.scalar<f32> {
    //            %result = rise.embed(%summand0, %summand1) {
    //                   %result = addf %summand0, %summand1 : f32
    //                   rise.return %result : f32
    //            } : !rise.scalar<f32>
    //            rise.return %result : !rise.scalar<f32>
    //        }
    //        %initializer = rise.literal #rise.lit<0.0>
    //        %reduce10Ints = rise.reduceSeq #rise.nat<4> #rise.scalar<f32> #rise.scalar<f32>
//    %reduce10Ints = rise.reduceSeq #rise.nat<4> #rise.scalar<f32> #rise.scalar<f32>
//    %result = rise.apply %reduce10Ints, %reductionAdd, %initializer, %multipliedArray

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

    OpBuilder builder(&expr);
    ScopedContext scope(builder, expr.getLoc());

    std::cout << "hier bin ich!\n" << std::flush;


//    Value newReduceApplication = reduceSeq(scalarF32Type(), [&](Value tuple, Value acc){
//      return (embed3(scalarF32Type(), ValueRange{fst(tuple),
//                                             snd(tuple), acc},
//                     [&](Value fst, Value snd, Value acc) {
//                       return acc * (fst + snd);
//                     }));
//    },initializer->getResult(0), mapInput);

    Value newReduceApplication = reduceSeq(scalarF32Type(), [&](Value y, Value acc){
          Value mapped = apply(scalarF32Type(), mapLambda->getResult(0), y);
          return apply(scalarF32Type(), reductionLambda->getResult(0), {mapped, acc});
    },initializer->getResult(0), mapInput);

    // cleanup
    expr.replaceAllUsesWith(newReduceApplication.getDefiningOp());
    expr.getParentOfType<FuncOp>().dump();

    expr.erase();
    reduction.erase();
//    reductionLambda->erase();
    reductionInput.erase();
    mapSeq.erase();
//    mapLambda->erase();

    std::cout << "hier bin ich!" << newReduceApplication.getDefiningOp()->getName().getStringRef().str() << "\n"  << std::flush;


    Operation *result = newReduceApplication.getDefiningOp();



    return success(*result);
  };
};

auto fuseReduceMap = []() { return FuseReduceMapStrategy(); };

#endif // LLVM_ELEVATE_ALGORITHMIC_H
