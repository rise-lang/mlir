//
// Created by martin on 30/10/2020.
//

#include "mlir/Dialect/Rise/Elevate2/algorithmic.h"
#include <mlir/Dialect/Rise/Elevate2/predicates.h>

using namespace mlir::rise;
using namespace mlir::edsc::abstraction;
using namespace mlir::edsc::op;
using namespace mlir::edsc::type;
using namespace mlir::elevate;

RewriteResult mlir::elevate::SplitJoinRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto applyMap = cast<ApplyOp>(op);

  if (!isa<MapSeqOp>(applyMap.getOperand(0).getDefiningOp())) return Failure();
  auto mapSeqOp = cast<MapSeqOp>(applyMap.getOperand(0).getDefiningOp());
  if (mapSeqOp.n().getIntValue() % n != 0) return Failure();
  // successful match

  auto mapLambda = applyMap.getOperand(1).getDefiningOp();
  Value mapInput = applyMap.getOperand(2);

  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(applyMap);

  Value result = join(mapSeq2D(mapSeqOp.t(), mapLambda->getResult(0), split(natType(n), mapInput)));

  return success(result.getDefiningOp());
}
auto mlir::elevate::splitJoin(const int n) -> SplitJoinRewritePattern {
  return SplitJoinRewritePattern(n);
}

RewriteResult mlir::elevate::FuseReduceMapRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto applyReduction = cast<ApplyOp>(op);

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

  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(applyReduction);

  Value newReduceApplication = reduceSeq(scalarF32Type(), [&](Value y, Value acc){
    Value mapped = apply(scalarF32Type(), mapLambda->getResult(0), y);
    return apply(scalarF32Type(), reductionLambda->getResult(0), {mapped, acc});
  },initializer->getResult(0), mapInput);

  Operation *result = newReduceApplication.getDefiningOp();
  return success(result);
}
auto mlir::elevate::fuseReduceMap() -> FuseReduceMapRewritePattern { return FuseReduceMapRewritePattern(); }

RewriteResult mlir::elevate::BetaReductionRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply = cast<ApplyOp>(op);
  if (!isa<LambdaOp>(apply.getOperand(0).getDefiningOp())) return Failure();
  // match success
  llvm::dbgs() << "doing betaRed now!\n";
  auto lambda = cast<LambdaOp>(apply.getOperand(0).getDefiningOp());

  SmallVector<Value, 10> args = SmallVector<Value, 10>();
  for (int i = 1; i < apply.getNumOperands(); i++) {
    args.push_back(apply.getOperand(i));
  }
  substitute(lambda, args);
  Value inlinedLambdaResult = inlineLambda(lambda, op->getBlock(), apply);
  Operation *result = inlinedLambdaResult.getDefiningOp();

  return success(result);
}
auto mlir::elevate::betaReduction() -> BetaReductionRewritePattern { return BetaReductionRewritePattern(); }

RewriteResult mlir::elevate::AddIdAfterRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply = cast<ApplyOp>(op);

  // successful match
  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(apply);

  auto newApply = rewriter.clone(*apply);
  Value result = mlir::edsc::op::id(newApply->getResult(0));

  return success(result.getDefiningOp());
}
auto mlir::elevate::addIdAfter() -> AddIdAfterRewritePattern { return AddIdAfterRewritePattern(); }

RewriteResult mlir::elevate::CreateTransposePairRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply = cast<ApplyOp>(op);
  if (!isa<IdOp>(apply.getOperand(0).getDefiningOp())) return Failure();
  auto id = cast<IdOp>(apply.getOperand(0).getDefiningOp());
  if (!apply.getResult().getType().isa<ArrayType>()) return Failure();
  if (!apply.getResult().getType().dyn_cast<ArrayType>().getElementType().isa<ArrayType>()) return Failure();

  // successful match
  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(apply);
  Value result = transpose(transpose(apply.getOperand(1)));

  return success(result.getDefiningOp());
}
auto mlir::elevate::createTransposePair() -> CreateTransposePairRewritePattern { return CreateTransposePairRewritePattern(); }

RewriteResult mlir::elevate::RemoveTransposePairRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply1 = cast<ApplyOp>(op);
  if (!isa<mlir::rise::TransposeOp>(apply1.getOperand(0).getDefiningOp())) return Failure();
  auto transpose1 = cast<mlir::rise::TransposeOp>(apply1.getOperand(0).getDefiningOp());
  if (!apply1.getOperand(1).isa<OpResult>()) return Failure();
  if (!isa<ApplyOp>(apply1.getOperand(1).getDefiningOp())) return Failure();
  auto apply2 = cast<ApplyOp>(apply1.getOperand(1).getDefiningOp());
  if (!isa<mlir::rise::TransposeOp>(apply2.getOperand(0).getDefiningOp())) return Failure();
  auto transpose2 = cast<mlir::rise::TransposeOp>(apply2.getOperand(0).getDefiningOp());
  if (!apply2.getOperand(1).isa<OpResult>()) return Failure();
  // successful match

  Value result = apply2.getOperand(1);

  return success(result.getDefiningOp());
}
auto mlir::elevate::removeTransposePair() -> RemoveTransposePairRewritePattern {return RemoveTransposePairRewritePattern(); }

RewriteResult mlir::elevate::MapMapFBeforeTransposeRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  // match for
  //    App(
  //        App(map(), lamA @ Lambda(_, App(
  //        App(map(), lamB @ Lambda(_, App(f, _))), _))),
  //    arg)

  if (!isa<ApplyOp>(op)) return Failure();
  auto apply1 = cast<ApplyOp>(op);
  if (!isa<mlir::rise::TransposeOp>(apply1.getOperand(0).getDefiningOp())) return Failure();
  auto transpose1 = cast<mlir::rise::TransposeOp>(apply1.getOperand(0).getDefiningOp());
  if (!apply1.getOperand(1).isa<OpResult>()) return Failure();
  if (!isa<ApplyOp>(apply1.getOperand(1).getDefiningOp())) return Failure();
  auto apply2 = cast<ApplyOp>(apply1.getOperand(1).getDefiningOp());
  llvm::dbgs() << "match 0!\n";

  //in the mm I am matching a join here bcs of splitJoin before. I prob. need to implement the RNF and DFNF for this
  if (!isa<MapSeqOp>(apply2.getOperand(0).getDefiningOp())) return Failure();
  llvm::dbgs() << "match 1!\n";

  auto mapSeqOp1 = cast<MapSeqOp>(apply2.getOperand(0).getDefiningOp());
  if (!apply2.getOperand(1).isa<OpResult>()) return Failure();
  if (!isa<LambdaOp>(apply2.getOperand(1).getDefiningOp())) return Failure();
  auto lamA = cast<LambdaOp>(apply2.getOperand(1).getDefiningOp()); // %2
  if (!isa<ApplyOp>(lamA.region().front().getTerminator()->getOperand(0).getDefiningOp())) return Failure();
  auto apply3 = cast<ApplyOp>(lamA.region().front().getTerminator()->getOperand(0).getDefiningOp());
  if (!isa<MapSeqOp>(apply3.getOperand(0).getDefiningOp())) return Failure();
  auto mapSeqOp2 = cast<MapSeqOp>(apply3.getOperand(0).getDefiningOp());
  if (!isa<LambdaOp>(apply3.getOperand(1).getDefiningOp())) return Failure();
  auto lamB = cast<LambdaOp>(apply3.getOperand(1).getDefiningOp()); // %9
  // successful match

  llvm::dbgs() << "match!\n";

  debug("lamA:")(lamA,rewriter);
  debug("lamB:")(lamB,rewriter);


  auto rrLamAEtaReducible = etaReducible()(lamA.getOperation(), rewriter);
  if (std::get_if<Failure>(&rrLamAEtaReducible)) return rrLamAEtaReducible;

  auto rrLamBEtaReducible = etaReducible()(lamB.getOperation(), rewriter);
  if (std::get_if<Failure>(&rrLamBEtaReducible)) return rrLamBEtaReducible;

  llvm::dbgs() << "Both lambdas are eta reducible\n";

  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(apply1);

//  apply1.getParentOfType<FuncOp>().dump();

  Operation *lambdaCopy = rewriter.clone(*lamB);

  Value result = mapSeq("scf", mapSeqOp1.t(), [&](Value elem){
    return mapSeq("scf", mapSeqOp2.t(), lambdaCopy->getResult(0), apply3.getOperand(2));
  }, transpose(apply2.getOperand(2)));

  return success(result.getDefiningOp());
}

auto mlir::elevate::mapMapFBeforeTranspose() -> MapMapFBeforeTransposeRewritePattern {
  return MapMapFBeforeTransposeRewritePattern();
}


//case e @ App(map(), Lambda(x, App(f, gx)))
//if !contains[Rise](x).apply(f) && !isIdentifier(gx) => // identifier are either results of in or a BlockArg
//gx.t match {
//    case _: DataType =>
//    Success((app(map, lambda(eraseType(x), gx)) >> map(f)) :: e.t)
//    case _ => Failure(mapLastFission())
//}
RewriteResult mlir::elevate::MapLastFissionRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto applyMap = cast<ApplyOp>(op);
  if (!isa<mlir::rise::MapSeqOp>(applyMap.getOperand(0).getDefiningOp())) return Failure();
  auto map = cast<mlir::rise::MapSeqOp>(applyMap.getOperand(0).getDefiningOp());
  if (!applyMap.getOperand(1).isa<OpResult>()) return Failure();
  if (!isa<LambdaOp>(applyMap.getOperand(1).getDefiningOp())) return Failure();
  LambdaOp lambda = cast<LambdaOp>(applyMap.getOperand(1).getDefiningOp());
//  llvm::dbgs() << "fission found applyMap\n";
  if (!isa<ApplyOp>(lambda.region().front().getTerminator()->getOperand(0).getDefiningOp())) return Failure();
  auto applyLastFunction = cast<ApplyOp>(lambda.region().front().getTerminator()->getOperand(0).getDefiningOp());
  auto f = applyLastFunction.getOperand(0).getDefiningOp();
  if (!applyLastFunction.getOperand(applyLastFunction.getNumOperands()-1).isa<OpResult>()) return Failure();
  if (!isa<ApplyOp>(applyLastFunction.getOperand(applyLastFunction.getNumOperands()-1).getDefiningOp())) return Failure();
  auto applyF = cast<ApplyOp>(applyLastFunction.getOperand(applyLastFunction.getNumOperands()-1).getDefiningOp());
  // match success
  llvm::dbgs() << "fission success\n";
  auto lastFunctionType = RiseDialect::getFunTypeOutput(lambda.getResult().getType().dyn_cast<FunType>());

  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(op);
//  RiseDialect::dumpRiseExpression(lambda.getParentOfType<FuncOp>());
//  lambda.getParentOfType<FuncOp>().dump();
//  f->dump();
//  applyLastFunction.dump();
//  applyF.dump();
//  RiseDialect::dumpRiseExpression(lambda);

//  applyF.getOperand(0).getDefiningOp()->dump(); // The Operands of this apply get messed up
  // TODO: go through the operands and check whether they depend on bla
  // Then replace that with the arg of the new lambda
//  llvm::dbgs() << "size: "  << lambda.region().front().getArguments().size() << (lambda.region().front().getArgument(0) == applyF.getOperand(2)) << "\n";
  applyLastFunction.replaceAllUsesWith(applyF.getResult()); // TODO:here the blockArg for zip is somehow messed up
//  llvm::dbgs() << "size: "  << lambda.region().front().getArguments().size() << (lambda.region().front().getArgument(0) == applyF.getOperand(2)) << "\n";

//  RiseDialect::dumpRiseExpression(lambda.getParentOfType<FuncOp>());
//  lambda.getParentOfType<FuncOp>().dump();
//  RiseDialect::dumpRiseExpression(lambda);

  // zip stays inside its mapSeq but its arguments are messed up. Check what the stuff below does. I think the mapSeq is actually replaced
//  lambda.dump();
//  RiseDialect::dumpRiseExpression(lambda);
  adjustLambdaType(lambda, rewriter);
//  RiseDialect::dumpRiseExpression(lambda);
//  lambda.dump();
//  llvm::dbgs() << "size: "  << lambda.region().front().getArguments().size() << (lambda.region().front().getArgument(0) == applyF.getOperand(2)) << "\n";

  // now build up new expr. with map
  auto newApplyMap =
      mapSeq("scf", RiseDialect::getAsDataType(applyF.getResult().getType()),
             lambda.getResult(), applyMap.getOperand(2));
//  RiseDialect::dumpRiseExpression(lambda.getParentOfType<FuncOp>());
//  llvm::dbgs() << "size: "  << lambda.region().front().getArguments().size() << (lambda.region().front().getArgument(0) == applyF.getOperand(2)) << "\n";
//  llvm::dbgs() << "size: "  << lambda.region().front().getArguments().size() << (lambda.region().front().getArgument(0) == lambda.region().front().getTerminator()->getOperand(0).getDefiningOp()->getOperand(2)) << "\n";

  auto result = mapSeq("scf", RiseDialect::getAsDataType(applyLastFunction.getResult().getType()), [&](Value elem){
        // clone function we want to move and the according apply
        // adjust operands on the apply to the new cloned function and the new input
//        f->moveAfter(rewriter.getBlock(), rewriter.getBlock()->front().getIterator());
    auto lastFunClone = scope.getBuilderRef().clone(*f);

        f->dump();
//        lastFunClone->dump(); // if this is a mapSeq, the lambda is not moved to it
        applyLastFunction.getOperation()->moveBefore(lastFunClone);
        lastFunClone->moveBefore(applyLastFunction);
//        lastFunClone->moveBefore()
//        auto newApply = rewriter.clone(*applyLastFunction);
        applyLastFunction.dump();
//    newApply->dump();

//        newApply->setOperand(0, lastFunClone->getResult(0));
//        newApply->setOperand(newApply->getNumOperands()-1, elem);
//        for (int i = 1; i < newApply->getNumOperands(); i++) {
//          if (!newApply->getOperand(i).isa<OpResult>()) continue;
//            newApply->getOperand(i).getDefiningOp()->moveBefore(lastFunClone);
//          newApply->getOperand(i).getDefiningOp()->dump();
//            newApply->dump();
//            // TODO: This does not always work!
//        }

    for (int i = 1; i < applyLastFunction.getNumOperands()-1; i++) {
      applyLastFunction.getOperand(i).dump();
      if (!applyLastFunction.getOperand(i).isa<OpResult>()) continue;
      applyLastFunction.getOperand(i).dump();
      llvm::dbgs() << "true";
      applyLastFunction.getOperand(i).getDefiningOp()->moveBefore(lastFunClone);
//      applyLastFunction.getOperand(i).getDefiningOp()->moveBefore(lambda.getParentOfType<FuncOp>().getRegion().front().getTerminator());


      //ERROR: TODO: For some reason this works perfectly for reduceSeq (moving lambda + literal)
      // But it does not work for mapSeq (moving lambda)

      // TODO: This does not always work!
    }
    applyLastFunction.setOperand(0, lastFunClone->getResult(0));
    applyLastFunction.setOperand(applyLastFunction.getNumOperands()-1, elem);

        // handle other operands, which may be out of scope.
        // TODO: maybe build a helper function to "inline" an operation
        // maybe we recursively have to inline other stuff.
        // see that it is inlined above the lastFunClone, bcs it has to be
        // defined be4 the apply

        return applyLastFunction.getResult();
  }, newApplyMap);

//  rewriter.eraseOp(applyLastFunction);
  rewriter.eraseOp(f);

//  debug("appLastFun")(applyLastFunction,rewriter);
//  debug("lastFun")(lastFunction,rewriter);

  return success(result.getDefiningOp());
}
auto mlir::elevate::mapLastFission() -> MapLastFissionRewritePattern {
  return MapLastFissionRewritePattern();
}

// utils
void mlir::elevate::substitute(LambdaOp lambda, llvm::SmallVector<Value, 10> args) {
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
mlir::Value mlir::elevate::inlineLambda(LambdaOp lambda, mlir::Block *insertionBlock, mlir::Operation *op) {
  mlir::Value lambdaResult =
      lambda.getRegion().front().getTerminator()->getOperand(0);
  insertionBlock->getOperations().splice(
      mlir::Block::iterator(op), lambda.getRegion().front().getOperations(),
      lambda.getRegion().front().begin(),
      mlir::Block::iterator(lambda.getRegion().front().getTerminator()));
  return lambdaResult;
}

// adjust the type of a lambda after changing its arguments or return value
void mlir::elevate::adjustLambdaType(LambdaOp lambda, PatternRewriter &rewriter) {
  FunType oldLambdaType = lambda.getResult().getType().dyn_cast<FunType>();
  if (!oldLambdaType) emitError(lambda.getLoc()) << "Lambda has inconsistent type!";
  Type outType = lambda.region().front().getTerminator()->getOperand(0).getType();
  auto argumentTypes = lambda.region().front().getArgumentTypes();

  FunType newFunType =
      FunType::get(rewriter.getContext(), (*llvm::make_reverse_iterator(argumentTypes.end())), outType);
  for (auto argType = llvm::make_reverse_iterator(argumentTypes.end());
       argType != llvm::make_reverse_iterator(argumentTypes.begin());
       argType++) {
    // inner nested funType is already built
    if (argType == llvm::make_reverse_iterator(argumentTypes.end())) continue;
    newFunType = FunType::get(rewriter.getContext(), *argType, newFunType);
  }
  lambda.getResult().setType(newFunType);
}