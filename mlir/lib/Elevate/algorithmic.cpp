//
// Created by martin on 30/10/2020.
//

#include "mlir/Dialect/Rise/Elevate2/algorithmic.h"
#include "mlir/Dialect/Rise/Elevate2/traversal.h"
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
  // match success - start rewriting

  auto mapLambda = applyMap.getOperand(1).getDefiningOp();
  Value mapInput = applyMap.getOperand(2);

  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(applyMap);

  Value result = join(mapSeq(arrayType(n, mapSeqOp.t()), [&](Value elem) {
        auto mapLambdaClone = scope.getBuilderRef().clone(*mapLambda);
        return mapSeq("scf", mapSeqOp.t(), mapLambdaClone->getResult(0), elem);
      }, split(natType(n), mapInput)));
  return success(result.getDefiningOp());
}
llvm::StringRef SplitJoinRewritePattern::getName() const {return llvm::StringRef("splitJoin");}
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
  auto mapSeq = cast<MapSeqOp>(reductionInput.getOperand(0).getDefiningOp());
  auto mapLambda = reductionInput.getOperand(1).getDefiningOp();
  Value mapInput = reductionInput.getOperand(2);
  // match success - start rewriting

  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(applyReduction);

  Value newReduceApplication = reduceSeq(scalarF32Type(), [&](Value y, Value acc){
    Value mapped = apply(scalarF32Type(), mapLambda->getResult(0), y);
    return apply(scalarF32Type(), reductionLambda->getResult(0), {mapped, acc});
  },initializer->getResult(0), mapInput);

  Operation *result = newReduceApplication.getDefiningOp();
  return success(result);
}
llvm::StringRef FuseReduceMapRewritePattern::getName() const {return llvm::StringRef("fuseReduceMap");}
auto mlir::elevate::fuseReduceMap() -> FuseReduceMapRewritePattern { return FuseReduceMapRewritePattern(); }

RewriteResult mlir::elevate::BetaReductionRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply = cast<ApplyOp>(op);
  if (!isa<LambdaOp>(apply.getOperand(0).getDefiningOp())) return Failure();
  // match success - start rewriting

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
llvm::StringRef BetaReductionRewritePattern::getName() const {return llvm::StringRef("betaReduction");}
auto mlir::elevate::betaReduction() -> BetaReductionRewritePattern { return BetaReductionRewritePattern(); }

RewriteResult mlir::elevate::AddIdAfterRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply = cast<ApplyOp>(op);

  // match success - start rewriting
  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(apply);

  auto newApply = rewriter.clone(*apply);
  Value result = mlir::edsc::op::id(newApply->getResult(0));

  return success(result.getDefiningOp());
}
llvm::StringRef AddIdAfterRewritePattern::getName() const {return llvm::StringRef("addId");}
auto mlir::elevate::addIdAfter() -> AddIdAfterRewritePattern { return AddIdAfterRewritePattern(); }

RewriteResult mlir::elevate::CreateTransposePairRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply = cast<ApplyOp>(op);
  if (!isa<IdOp>(apply.getOperand(0).getDefiningOp())) return Failure();
  auto id = cast<IdOp>(apply.getOperand(0).getDefiningOp());
  if (!apply.getResult().getType().isa<ArrayType>()) return Failure();
  if (!apply.getResult().getType().dyn_cast<ArrayType>().getElementType().isa<ArrayType>()) return Failure();

  // match success - start rewriting
  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(apply);
  Value result = transpose(transpose(apply.getOperand(1)));

  return success(result.getDefiningOp());
}
llvm::StringRef CreateTransposePairRewritePattern::getName() const {return llvm::StringRef("createTransposePair");}
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
  // match success - start rewriting

  Value result = apply2.getOperand(1);

  return success(result.getDefiningOp());
}
llvm::StringRef RemoveTransposePairRewritePattern::getName() const {return llvm::StringRef("removeTransposePair");}
auto mlir::elevate::removeTransposePair() -> RemoveTransposePairRewritePattern {return RemoveTransposePairRewritePattern(); }

// match for
//    App(
//        App(map(), lamA @ Lambda(_, App(
//        App(map(), lamB @ Lambda(_, App(f, _))), _))),
//    arg)
RewriteResult mlir::elevate::MoveMapMapFBeforeTransposeRewritePattern::impl(Operation *op, PatternRewriter &rewriter) const {
  if (!isa<ApplyOp>(op)) return Failure();
  auto apply1 = cast<ApplyOp>(op);
  if (!isa<mlir::rise::TransposeOp>(apply1.getOperand(0).getDefiningOp())) return Failure();
  auto transpose1 = cast<mlir::rise::TransposeOp>(apply1.getOperand(0).getDefiningOp());
  if (!apply1.getOperand(1).isa<OpResult>()) return Failure();
  if (!isa<ApplyOp>(apply1.getOperand(1).getDefiningOp())) return Failure();
  auto apply2 = cast<ApplyOp>(apply1.getOperand(1).getDefiningOp());
  if (!isa<MapSeqOp>(apply2.getOperand(0).getDefiningOp())) return Failure();
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
  // match success - start rewriting

  // TODO: is etareducible check required in mlir rise?
  //  auto rrLamAEtaReducible = etaReducible()(lamA.getOperation(), rewriter);
  //  if (std::get_if<Failure>(&rrLamAEtaReducible)) return rrLamAEtaReducible;
  //  auto rrLamBEtaReducible = etaReducible()(lamB.getOperation(), rewriter);
  //  if (std::get_if<Failure>(&rrLamBEtaReducible)) return rrLamBEtaReducible;

  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(apply1);

  Value transposedInput = transpose(apply2.getOperand(2));
  Value result = mapSeq("scf",  arrayType(transposedInput.getType().cast<ArrayType>().getElementType().cast<ArrayType>().getSize(), mapSeqOp2.t()) , [&](Value elem) {
    Operation *lambdaCopy = rewriter.clone(*lamB);
    return mapSeq("scf", mapSeqOp2.t(), lambdaCopy->getResult(0), elem);//apply3.getOperand(2));
  }, transposedInput);

  return success(result.getDefiningOp());
}
llvm::StringRef MoveMapMapFBeforeTransposeRewritePattern::getName() const {return llvm::StringRef("moveMapMapFBeforeTranspose");}
auto mlir::elevate::moveMapMapFBeforeTranspose() -> MoveMapMapFBeforeTransposeRewritePattern {
  return MoveMapMapFBeforeTransposeRewritePattern();
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

  if (!isa<ApplyOp>(lambda.region().front().getTerminator()->getOperand(0).getDefiningOp())) return Failure();
  auto applyLastFunction = cast<ApplyOp>(lambda.region().front().getTerminator()->getOperand(0).getDefiningOp());
  auto f = applyLastFunction.getOperand(0).getDefiningOp();

  // use contains to check that f does not use blockarg
  for (int i = 0; i < applyLastFunction->getNumOperands()-1; i++) {
    if (!applyLastFunction.getOperand(i).isa<OpResult>()) {
      applyLastFunction.getOperand(i).dump();
      continue;
    }
    auto res = contains(lambda.getRegion().front().getArgument(0))(applyLastFunction.getOperand(i).getDefiningOp(), rewriter);
    if (std::get_if<Success>(&res)) {
      return Failure();
    }
  }

  if (!applyLastFunction.getOperand(applyLastFunction.getNumOperands()-1).isa<OpResult>()) return Failure();
  if (!isa<ApplyOp>(applyLastFunction.getOperand(applyLastFunction.getNumOperands()-1).getDefiningOp())) return Failure();
  auto applyG = cast<ApplyOp>(applyLastFunction.getOperand(applyLastFunction.getNumOperands()-1).getDefiningOp());

  // match success - start rewriting
  ScopedContext scope(rewriter, op->getLoc());
  rewriter.setInsertionPointAfter(op);
  applyLastFunction.replaceAllUsesWith(applyG.getResult()); // TODO:here the blockArg for zip is somehow messed up
  adjustLambdaType(lambda, rewriter);

  // build up new expr. with map
  auto newApplyMap =
      mapSeq("scf", RiseDialect::getAsDataType(applyG.getResult().getType()),
             lambda.getResult(), applyMap.getOperand(2));

  auto result = mapSeq("scf", RiseDialect::getAsDataType(applyLastFunction.getResult().getType()), [&](Value elem) {
    // clone function and the according apply we want to move
    auto lastFunClone = scope.getBuilderRef().clone(*f);
    applyLastFunction.getOperation()->moveBefore(lastFunClone);
    lastFunClone->moveBefore(applyLastFunction);

    // adjust operands on the apply to the new cloned function and the new input
    for (int i = 1; i < applyLastFunction.getNumOperands()-1; i++) {
      if (!applyLastFunction.getOperand(i).isa<OpResult>()) continue;
      applyLastFunction.getOperand(i).getDefiningOp()->moveBefore(lastFunClone);
    }
    applyLastFunction.setOperand(0, lastFunClone->getResult(0));
    applyLastFunction.setOperand(applyLastFunction.getNumOperands()-1, elem);

    return applyLastFunction.getResult();
  }, newApplyMap);

  rewriter.eraseOp(f);
  return success(result.getDefiningOp());
}
llvm::StringRef MapLastFissionRewritePattern::getName() const {return llvm::StringRef("mapLastFission");}
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