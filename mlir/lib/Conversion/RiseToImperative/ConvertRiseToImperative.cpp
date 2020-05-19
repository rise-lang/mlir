//
// Created by martin on 26/11/2019.
//

#include "mlir/Conversion/RiseToImperative/ConvertRiseToImperative.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/EDSC/Builders.h>

using namespace mlir;
using namespace mlir::rise;

namespace {
struct ConvertRiseToImperativePass
    : public RiseToImperativeBase<ConvertRiseToImperativePass> {
  void runOnFunction() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//
struct RiseToImperativePattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;
  LogicalResult match(FuncOp funcOp) const override;
  void rewrite(FuncOp funcOp, PatternRewriter &rewriter) const override;
  //  PatternMatchResult matchAndRewrite(FuncOp riseFun,
  //                                     PatternRewriter &rewriter) const
  //                                     override;
};

LogicalResult RiseToImperativePattern::match(FuncOp funcOp) const {
  bool riseInside = false;
  //  std::cout << "matching! \n" << std::flush;

  if (funcOp.isExternal())
    return failure();
  // I currently check whether a rise.in is inside the func.
  funcOp.walk([&](Operation *op) {
    //    if (op->getDialect()->getNamespace().equals(
    //        rise::RiseDialect::getDialectNamespace()))
    if (isa<RiseInOp>(op))
      riseInside = true;
  });

  if (riseInside) {
    //    std::cout << "matchSuccess! \n" << std::flush;

    return success();
  } else {
    return failure();
  }
}

void RiseToImperativePattern::rewrite(FuncOp funcOp,
                                      PatternRewriter &rewriter) const {
  MLIRContext *context = rewriter.getContext();
  Block &block = funcOp.getBody().front();

  rewriter.setInsertionPointToStart(&funcOp.getBody().front());
  funcOp.walk([&](Operation *op) {
    if (RiseInOp inOp = dyn_cast<RiseInOp>(op)) {
      inOp.output().replaceAllUsesWith(inOp.input());
      // TODO: errors could come from this
      //      inOp.erase();
      //      rewriter.eraseOp(inOp);
      rewriter.setInsertionPointAfter(inOp);
    }
  });

  // Start at the back and find the first apply.
  ApplyOp applyOp;
  for (auto op = block.rbegin(); op != block.rend(); op++) {
    if (isa<ApplyOp>(*op)) {
      applyOp = cast<ApplyOp>(*op);
      break;
    }
  }

  // Translation to imperative
  AccT(applyOp, funcOp.getBody().front().getArgument(0), rewriter);

  emitRemark(funcOp.getLoc()) << "AccT finished. Starting CodeGen.";

  SmallVector<rise::RiseAssignOp, 10> assignOps;
  funcOp.walk([&assignOps](Operation *inst) {
    if (!inst)
      return;
    if (isa<RiseAssignOp>(inst)) {
      assignOps.push_back(cast<RiseAssignOp>(inst));
    }
    return;
  });

  // Codegen:
  bool doCodegen = true;
  SmallVector<Operation *, 10> erasureList = {};
  if (doCodegen) {
    for (rise::RiseAssignOp assign : assignOps) {
      codeGen(assign, {}, rewriter);
    }
  }
  emitRemark(funcOp.getLoc()) << "CodeGen finished. Starting Cleanup.";

  //
  //  //   cleanup:
  //  //   erase intermediate operations.
  //  //   We remove them back to front right now,
  //  //   this should be alright in terms of dependencies.
  //  // TODO:
  //  //  hack here: I also remove lambdas and applies and wraps which should
  //  //  have been removed in the AccT. I could not remove them there
  //  presumably
  //  //  due to a bug with erasure notifications in MLIR. Later check whether
  //  this
  //  //  has been fixed already.
  if (doCodegen) {
    funcOp.walk([&](Operation *inst) {
      if (inst->getDialect()->getNamespace().equals("rise")) {
        if (inst->getParentOfType<LambdaOp>()) {
          //          std::cout << "not erasing: " <<
          //          inst->getName().getStringRef().str()
          //                    << "\n"
          //                    << std::flush;
        } else {
          erasureList.push_back(inst);
        }
      }
      return;
    });
  } else {
    funcOp.walk([&erasureList](Operation *inst) {
      if (isa<ApplyOp>(inst) || isa<LambdaOp>(inst) || isa<RiseInOp>(inst) ||
          isa<MapSeqOp>(inst)) {
        if (inst->getParentOfType<LambdaOp>()) {
          std::cout << "not erasing: " << inst->getName().getStringRef().str()
                    << "\n"
                    << std::flush;
        } else {
          erasureList.push_back(inst);
        }
      }
      return;
    });
  }

  // Operations which are inside a lambda are erased twice
  size_t unneededOps = erasureList.size();
  for (size_t i = 0; i < unneededOps; i++) {
    auto op = erasureList.pop_back_val();
    //    std::cout << "erasing: " << op->getName().getStringRef().str() << "\n"
    //              << std::flush;
    op->dropAllUses();
    op->dropAllReferences();
    rewriter.eraseOp(op);
  }

  return;
}
// std::cout << "\n" << "" << "\n" << std::flush;

/// Acceptor Translation
/// apply - The ApplyOp to start the translation.
/// outsideArgs - any mlir::Value which are operands of rise.fun or a lambda

// TODO: continue here!
//       AccT should not take a Lambda anymore but prob. just an Operation.
//       maybe always a rise.return? Then from there check whether we have an
//       apply
//    -> continue as before.
//       or have a wrap operation
//    -> stuff between the wraps and unwraps has to be copied into the
//    translated stuff.

void mlir::rise::AccT(ReturnOp returnOp, Value out, PatternRewriter &rewriter) {
  if (!returnOp.getOperand(0).isa<OpResult>()) {
    emitRemark(returnOp.getLoc())
        << "Directly returning an argument is not supported in lowering to "
           "imperative currently";
    return;
  }
  if (ApplyOp apply =
          dyn_cast<ApplyOp>(returnOp.getOperand(0).getDefiningOp())) {
    AccT(apply, out, rewriter);
    return;
  } else if (RiseEmbedOp embedOp = dyn_cast<RiseEmbedOp>(
                 returnOp.getOperand(0).getDefiningOp())) {
    emitRemark(returnOp.getLoc())
        << "AccT of RiseEmbedOp. Copy operations from this block to result.";

    // Translating all operands first
    //    for (auto operand = embedOp.getOperands().begin(); operand !=
    //    embedOp.getOperands().end(); operand++) {
    for (int i = 0; i < embedOp.getOperands().size(); i++) {
      auto operand = embedOp.getOperand(i);
      auto operandCont = ConT(operand, rewriter.getInsertionPoint(), rewriter);
      embedOp.region().front().getArgument(i).replaceAllUsesWith(operandCont);
    }
    rise::ReturnOp embedReturn = dyn_cast<rise::ReturnOp>(
        embedOp.getRegion().front().getOperations().back());

    rewriter.getInsertionBlock()->getOperations().splice(
        rewriter.getInsertionPoint(),
        embedOp.getRegion().front().getOperations(),
        embedOp.getRegion().front().begin(), Block::iterator(embedReturn));

    auto assignment = rewriter.create<RiseAssignOp>(
        embedReturn.getLoc(), embedReturn.getOperand(0), out);

    //    rewriter.eraseOp(embedOp);
    return;

  } else {
    std::cout << "something went wrong! \n" << std::flush;
    return;
  }
}

void mlir::rise::AccT(ApplyOp apply, Value out, PatternRewriter &rewriter) {
  // lower outsideArgs first

  Operation *appliedFun = apply.getOperand(0).getDefiningOp();
  OpBuilder::InsertPoint savedInsertionPoint = rewriter.saveInsertionPoint();
  // If functions are applied partially i.e. appliedFun is an ApplyOp we
  // iterate here until we reach a non ApplyOp We push the operands of the
  // applies into a vector, as only the effectively applied Op knows what to
  // do with them. We always leave out operand 0, as this is the applied
  // function Operands in the Vector will be ordered such that the top most
  // operand is the first needed to the applied function. As applies are
  // processed bottom-up, in the case that an apply has more than 1 operand
  // they have to be added right to left to keep order.
  SmallVector<Value, 10> applyOperands = SmallVector<Value, 10>();
  SmallVector<ApplyOp, 10> applyStack = SmallVector<ApplyOp, 10>();
  applyStack.push_back(apply);
  for (int i = apply.getNumOperands() - 1; i > 0; i--) {
    applyOperands.push_back(apply.getOperand(i));
  }
  while (auto next_apply = dyn_cast<ApplyOp>(appliedFun)) {
    apply = next_apply;
    appliedFun = apply.getOperand(0).getDefiningOp();
    applyStack.push_back(apply);

    for (int i = apply.getNumOperands() - 1; i > 0; i--) {
      applyOperands.push_back(apply.getOperand(i));
    }
  }

  Location loc = apply.getLoc();
  if (isa<ReduceSeqOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of ReduceSeq";

    auto n = appliedFun->getAttrOfType<NatAttr>("n");
    auto s = appliedFun->getAttrOfType<DataTypeAttr>("s");
    auto t = appliedFun->getAttrOfType<DataTypeAttr>("t");

    auto reductionFun = applyOperands.pop_back_val();

    auto initializer = applyOperands.pop_back_val();
    auto array = applyOperands.pop_back_val();

    // Add Continuation for array.
    auto contArray = ConT(array, rewriter.getInsertionPoint(), rewriter);

    auto cst_zero = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 0);

    bool defineNewAccumulator = false;
    // Accumulator for Reduction
    // TODO: do this properly! This can be way better structured
    Value accum;
    if (defineNewAccumulator) {
      auto contInit = ConT(initializer, rewriter.getInsertionPoint(), rewriter);

      accum = rewriter
                  .create<AllocOp>(
                      appliedFun->getLoc(),
                      MemRefType::get(ArrayRef<int64_t>{1},
                                      FloatType::getF32(rewriter.getContext())))
                  .getResult();

      rewriter.create<linalg::FillOp>(initializer.getLoc(), accum, contInit);
    } else {
      auto contInit = ConT(initializer, rewriter.getInsertionPoint(), rewriter);
      accum = out;
      auto accumIdx = rewriter.create<RiseIdxOp>(
          accum.getLoc(), FloatType::getF32(rewriter.getContext()), accum,
          cst_zero.getResult());
      auto initAccum = rewriter.create<RiseAssignOp>(
          appliedFun->getLoc(), contInit, accumIdx.getResult());
    }
    // zero constant for indexing

    Value loopInductionVar;
    Block *forLoopBody;

    // lowering to a specific loop depending on the lowering target dialect
    std::string loweringTarget;
    if (StringAttr loweringTargetAttr =
            appliedFun->getAttrOfType<StringAttr>("to")) {
      loweringTarget = loweringTargetAttr.getValue().str();
    } else {
      // default lowering target
      loweringTarget = "loop";
    }

    if (loweringTarget == "loop") {
      auto lowerBound =
          rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 0);
      auto upperBound = rewriter.create<ConstantIndexOp>(
          appliedFun->getLoc(), n.getValue().getIntValue());
      auto step = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 1);

      auto forLoop =
          rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
      loopInductionVar = forLoop.getInductionVar();
      forLoopBody = forLoop.getBody();
    } else if (loweringTarget == "affine") {
      auto forLoop =
          rewriter.create<AffineForOp>(loc, 0, n.getValue().getIntValue(), 1);
      loopInductionVar = forLoop.getInductionVar();
      forLoopBody = forLoop.getBody();
    }

    rewriter.setInsertionPointToStart(forLoopBody);
    LambdaOp reductionLambda = dyn_cast<LambdaOp>(reductionFun.getDefiningOp());
    //    if (!reductionLambda) {
    //      reductionLambda = expandToLambda(reductionFun, rewriter);
    //    }

    RiseIdxOp xi;

    // index into input
    if (contArray.getType().isa<MemRefType>()) {
      ArrayRef<int64_t> inShape =
          contArray.getType().dyn_cast<MemRefType>().getShape().drop_back(1);
      Type inIndexOpResult;
      if (inShape.size() > 0) {
        inIndexOpResult =
            MemRefType::get(inShape, FloatType::getF32(rewriter.getContext()));
      } else {
        inIndexOpResult = FloatType::getF32(rewriter.getContext());
      }
      xi = rewriter.create<RiseIdxOp>(loc, inIndexOpResult, contArray,
                                      loopInductionVar);
      //      std::cout << "contArray is of MemrefType";
    } else if (isa<RiseIdxOp>(contArray.getDefiningOp())) {
      //      std::cout << "contArray is not of MemrefType";
    }

    RiseIdxOp accumIdx;
    if (defineNewAccumulator) {
      // index into acc
      accumIdx = rewriter.create<RiseIdxOp>(
          accum.getLoc(), FloatType::getF32(rewriter.getContext()), accum,
          cst_zero.getResult());
    } else {
      accumIdx = rewriter.create<RiseIdxOp>(
          accum.getLoc(), FloatType::getF32(rewriter.getContext()), accum,
          cst_zero.getResult());
      //      ArrayRef<int64_t> outShape =
      //          out.getType().dyn_cast<MemRefType>().getShape();
      //      MemRefType outIndexOpResult =
      //          MemRefType::get(outShape,
      //          FloatType::getF32(rewriter.getContext()));
      //      // TODO: at this point we sometimes need the 0 to access a val and
      //      // sometimes not.
      //      accumIdx = rewriter.create<RiseIdxOp>(loc, outIndexOpResult, out,
      //                                            cst_zero.getResult());
    }
    // operate on a copy of the lambda to avoid generating dependencies.
    LambdaOp lambdaCopy = cast<LambdaOp>(rewriter.clone(*reductionLambda));
    auto fxi = rewriter.create<ApplyOp>(
        loc, lambdaCopy.getType(), lambdaCopy.getResult(),
        ValueRange{accumIdx.getResult(), xi.getResult()});

    AccT(fxi, accumIdx.getResult(), rewriter);

    //        fxi.getResult().dropAllUses();
    //        fxi.erase();
    //    ////
    //    lambdaCopy.getResult().dropAllUses();
    //    rewriter.eraseOp(lambdaCopy);

    // copy accumulator to output
    if (defineNewAccumulator) {
      rewriter.setInsertionPointAfter(forLoopBody->getParentOp());

      ArrayRef<int64_t> outShape =
          out.getType().dyn_cast<MemRefType>().getShape();
      MemRefType outIndexOpResult =
          MemRefType::get(outShape, FloatType::getF32(rewriter.getContext()));
      // TODO: at this point we sometimes need the 0 to access a val and
      // sometimes not.
      auto out0 = rewriter.create<RiseIdxOp>(loc, outIndexOpResult, out,
                                             cst_zero.getResult());

      RiseIdxOp storeAccIdx = rewriter.create<RiseIdxOp>(
          accum.getLoc(),
          MemRefType::get({1}, FloatType::getF32(rewriter.getContext())), accum,
          cst_zero.getResult());

      auto storing = rewriter.create<RiseAssignOp>(
          appliedFun->getLoc(), storeAccIdx.getResult(), out0.getResult());

      rewriter.restoreInsertionPoint(savedInsertionPoint);
    }

    return;

  } else if (isa<MapSeqOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of MapSeq";

    //     For now we treat all maps as mapSeqs
    auto n = appliedFun->getAttrOfType<NatAttr>("n");
    auto s = appliedFun->getAttrOfType<DataTypeAttr>("s");
    auto t = appliedFun->getAttrOfType<DataTypeAttr>("t");

    auto f = applyOperands.pop_back_val();

    auto array = applyOperands.pop_back_val();

    auto contArray = ConT(array, rewriter.getInsertionPoint(), rewriter);

    // zero constant for indexing
    auto cst_zero = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 0);

    Value loopInductionVar;
    Block *forLoopBody;

    // lowering to a specific loop depending on the lowering target dialect
    std::string loweringTarget;
    if (StringAttr loweringTargetAttr =
            appliedFun->getAttrOfType<StringAttr>("to")) {
      loweringTarget = loweringTargetAttr.getValue().str();
    } else {
      // default lowering target
      loweringTarget = "loop";
    }

    if (loweringTarget == "loop") {
      auto lowerBound =
          rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 0);
      auto upperBound = rewriter.create<ConstantIndexOp>(
          appliedFun->getLoc(), n.getValue().getIntValue());
      auto step = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 1);

      auto forLoop = rewriter.create<mlir::scf::ForOp>(
          appliedFun->getLoc(), lowerBound, upperBound, step);
      loopInductionVar = forLoop.getInductionVar();
      forLoopBody = forLoop.getBody();
    } else if (loweringTarget == "affine") {
      auto forLoop = rewriter.create<AffineForOp>(
          appliedFun->getLoc(), 0, n.getValue().getIntValue(), 1);
      loopInductionVar = forLoop.getInductionVar();
      forLoopBody = forLoop.getBody();
    }

    rewriter.setInsertionPointToStart(forLoopBody);

    LambdaOp fLambda = dyn_cast<LambdaOp>(f.getDefiningOp());
    //    if (!fLambda) {
    //      fLambda = expandToLambda(f, rewriter);
    //    }

    RiseIdxOp xi;
    if (contArray.getType().isa<MemRefType>()) {
      ArrayRef<int64_t> inShape =
          contArray.getType().dyn_cast<MemRefType>().getShape().drop_back(1);

      MemRefType inIndexOpResult =
          MemRefType::get(inShape, FloatType::getF32(rewriter.getContext()));

      xi = rewriter.create<RiseIdxOp>(loc, inIndexOpResult, contArray,
                                      loopInductionVar);
    } else if (isa<RiseIdxOp>(contArray.getDefiningOp())) {
      //      std::cout << "got an idx and not a memref! \n" << std::flush;
    }

    // operate on a copy of the lambda to avoid generating dependencies.
    LambdaOp lambdaCopy = cast<LambdaOp>(rewriter.clone(*fLambda));
    auto fxi = rewriter.create<ApplyOp>(loc, lambdaCopy.getType(),
                                        lambdaCopy.getResult(), xi.getResult());

    ArrayRef<int64_t> outShape =
        contArray.getType().dyn_cast<MemRefType>().getShape().drop_back(1);
    MemRefType outIndexOpResult =
        MemRefType::get(outShape, FloatType::getF32(rewriter.getContext()));
    auto outi = rewriter.create<RiseIdxOp>(loc, outIndexOpResult, out,
                                           loopInductionVar);

    AccT(fxi, outi.getResult(), rewriter);
    // tmp Apply not needed anymore.

    //    //    std::cout << "deleting apply dependency for lambda" <<
    //    std::flush; fxi.getResult().dropAllUses(); fxi.erase();
    //    //    rewriter.eraseOp(fxi);
    //
    //    lambdaCopy.getResult().dropAllUses();
    //    rewriter.eraseOp(lambdaCopy);

    rewriter.setInsertionPointAfter(forLoopBody->getParentOp());
    return;
  } else if (isa<MapParOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of MapPar";

    //     For now we treat all maps as mapSeqs
    auto n = appliedFun->getAttrOfType<NatAttr>("n");
    auto s = appliedFun->getAttrOfType<DataTypeAttr>("s");
    auto t = appliedFun->getAttrOfType<DataTypeAttr>("t");

    auto f = applyOperands.pop_back_val();

    auto array = applyOperands.pop_back_val();

    auto contArray = ConT(array, rewriter.getInsertionPoint(), rewriter);

    Value loopInductionVar;
    Block *forLoopBody;

    // lowering to a specific loop depending on the lowering target dialect
    std::string loweringTarget;
    if (StringAttr loweringTargetAttr =
            appliedFun->getAttrOfType<StringAttr>("to")) {
      loweringTarget = loweringTargetAttr.getValue().str();
    } else {
      // default lowering target
      loweringTarget = "loop";
    }

    // TODO: These have no parallel semantics yet
    if (loweringTarget == "loop") {
      auto lowerBound =
          rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 0);
      auto upperBound = rewriter.create<ConstantIndexOp>(
          appliedFun->getLoc(), n.getValue().getIntValue());
      auto step = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 1);

      auto forLoop = rewriter.create<mlir::scf::ForOp>(
          appliedFun->getLoc(), lowerBound, upperBound, step);
      loopInductionVar = forLoop.getInductionVar();
      forLoopBody = forLoop.getBody();
    } else if (loweringTarget == "affine") {
      auto forLoop = rewriter.create<AffineForOp>(
          appliedFun->getLoc(), 0, n.getValue().getIntValue(), 1);

      // TODO: Not working as intended
      //      auto lbMap = AffineMap::getConstantMap(0, rewriter.getContext());
      //      auto ubMap = AffineMap::getConstantMap(n.getValue().getIntValue(),
      //                                             rewriter.getContext());
      //
      //      auto parallelOp =
      //      rewriter.create<AffineParallelOp>(appliedFun->getLoc(),
      //                                                          lbMap,
      //                                                          ValueRange{},
      //                                                          ubMap,
      //                                                          ValueRange{});
      //      loopInductionVar = *parallelOp.getLowerBoundsOperands().begin();
      //      forLoopBody = parallelOp.getBody();
      loopInductionVar = forLoop.getInductionVar();
      forLoopBody = forLoop.getBody();
    }

    rewriter.setInsertionPointToStart(forLoopBody);

    LambdaOp fLambda = dyn_cast<LambdaOp>(f.getDefiningOp());
    //    if (!fLambda) {
    //      fLambda = expandToLambda(f, rewriter);
    //    }

    RiseIdxOp xi;
    if (contArray.getType().isa<MemRefType>()) {
      ArrayRef<int64_t> inShape =
          contArray.getType().dyn_cast<MemRefType>().getShape().drop_back(1);

      MemRefType inIndexOpResult =
          MemRefType::get(inShape, FloatType::getF32(rewriter.getContext()));

      xi = rewriter.create<RiseIdxOp>(loc, inIndexOpResult, contArray,
                                      loopInductionVar);
    } else if (isa<RiseIdxOp>(contArray.getDefiningOp())) {
      //      std::cout << "got an idx and not a memref! \n" << std::flush;
    }

    // operate on a copy of the lambda to avoid generating dependencies.
    LambdaOp lambdaCopy = cast<LambdaOp>(rewriter.clone(*fLambda));
    auto fxi = rewriter.create<ApplyOp>(loc, lambdaCopy.getType(),
                                        lambdaCopy.getResult(), xi.getResult());

    ArrayRef<int64_t> outShape =
        contArray.getType().dyn_cast<MemRefType>().getShape().drop_back(1);
    MemRefType outIndexOpResult =
        MemRefType::get(outShape, FloatType::getF32(rewriter.getContext()));
    auto outi = rewriter.create<RiseIdxOp>(loc, outIndexOpResult, out,
                                           loopInductionVar);

    AccT(fxi, outi.getResult(), rewriter);
    // tmp Apply not needed anymore.

    //    fxi.getResult().dropAllUses();
    //    fxi.erase();
    //    //    rewriter.eraseOp(fxi);
    //
    //    lambdaCopy.getResult().dropAllUses();
    //    rewriter.eraseOp(lambdaCopy);

    rewriter.setInsertionPointAfter(forLoopBody->getParentOp());
    return;
  } else if (FstOp fstOp = dyn_cast<FstOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Fst";

    auto tuple = applyOperands.pop_back_val();
    auto contTuple = ConT(tuple, rewriter.getInsertionPoint(), rewriter);

    auto fstIntermOp = rewriter.create<RiseFstIntermediateOp>(
        fstOp.getLoc(), FloatType::getF32(rewriter.getContext()), contTuple);
    auto assignment = rewriter.create<RiseAssignOp>(
        appliedFun->getLoc(), fstIntermOp.getResult(), out);
    return;
  } else if (SndOp sndOp = dyn_cast<SndOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Snd";

    auto tuple = applyOperands.pop_back_val();
    auto contTuple = ConT(tuple, rewriter.getInsertionPoint(), rewriter);

    auto sndIntermOp = rewriter.create<RiseSndIntermediateOp>(
        sndOp.getLoc(), FloatType::getF32(rewriter.getContext()), contTuple);
    auto assignment = rewriter.create<RiseAssignOp>(
        appliedFun->getLoc(), sndIntermOp.getResult(), out);
    return;
  } else if (isa<LambdaOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Lambda";

    LambdaOp lambda = cast<LambdaOp>(appliedFun);
    Substitute(lambda, applyOperands);

    // Find return in Lambda Region to start new AccT
    rise::ReturnOp returnOp;
    for (auto op = lambda.region().front().rbegin();
         op != lambda.region().front().rend(); op++) {
      if (isa<ReturnOp>(*op)) {
        returnOp = cast<ReturnOp>(*op);
        break;
      }
    }
    AccT(returnOp, out, rewriter);
    return;
  } else if (isa<AddOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Add";

    auto summand0 = applyOperands.pop_back_val();
    auto summand1 = applyOperands.pop_back_val();

    auto contSummand0 = ConT(summand0, rewriter.getInsertionPoint(), rewriter);
    auto contSummand1 = ConT(summand1, rewriter.getInsertionPoint(), rewriter);

    //    auto newAddOp = rewriter.create<AddFOp>(appliedFun->getLoc(),
    //    contSummand0,
    //                                            contSummand1);
    auto newAddOp = rewriter.create<RiseBinaryOp>(
        appliedFun->getLoc(), FloatType::getF32(rewriter.getContext()),
        StringAttr::get("add", rewriter.getContext()), contSummand0,
        contSummand1);
    auto assignment = rewriter.create<RiseAssignOp>(appliedFun->getLoc(),
                                                    newAddOp.getResult(), out);

    return;
  } else if (isa<MulOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Mul";

    auto factor0 = applyOperands.pop_back_val();
    auto factor1 = applyOperands.pop_back_val();

    auto contFactor0 = ConT(factor0, rewriter.getInsertionPoint(), rewriter);
    auto contFactor1 = ConT(factor1, rewriter.getInsertionPoint(), rewriter);

    auto newMulOp = rewriter.create<RiseBinaryOp>(
        appliedFun->getLoc(), FloatType::getF32(rewriter.getContext()),
        StringAttr::get("mul", rewriter.getContext()), contFactor0,
        contFactor1);

    auto assignment = rewriter.create<RiseAssignOp>(appliedFun->getLoc(),
                                                    newMulOp.getResult(), out);

    return;
  } else if (isa<ApplyOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Apply";

    emitError(appliedFun->getLoc()) << "We should never get here";
  } else {
    emitRemark(appliedFun->getLoc())
        << "lowering op: " << appliedFun->getName() << " not yet supported.";
  }
}

/// Continuation Translation
mlir::Value mlir::rise::ConT(mlir::Value contValue,
                             Block::iterator contLocation,
                             PatternRewriter &rewriter) {

  Location loc = contValue.getLoc();
  auto oldInsertPoint = rewriter.saveInsertionPoint();

  if (contValue.isa<OpResult>()) {
    if (isa<LiteralOp>(contValue.getDefiningOp())) {
      emitRemark(contValue.getLoc()) << "ConT of Literal";
      std::string literalValue = dyn_cast<LiteralOp>(contValue.getDefiningOp())
                                     .literalAttr()
                                     .getValue();

      emitRemark(contValue.getLoc()) << "Literal value: " << literalValue;

      if (LiteralOp op = dyn_cast<LiteralOp>(contValue.getDefiningOp())) {
        if (op.literalAttr()
                .getType()
                .isa<ScalarType>()) { // TODO: use contained type for generating
                                      // this
          auto fillOp = rewriter.create<ConstantFloatOp>(
              loc, llvm::APFloat(std::stof(literalValue)),
              FloatType::getF32(rewriter.getContext()));

          rewriter.restoreInsertionPoint(oldInsertPoint);
          return fillOp.getResult();
        } else if (ArrayType arrayType =
                       op.literalAttr().getType().dyn_cast<ArrayType>()) {
          SmallVector<int64_t, 4> shape = {};

          shape.push_back(arrayType.getSize().getIntValue());
          while (arrayType.getElementType().isa<ArrayType>()) {
            arrayType = arrayType.getElementType().dyn_cast<ArrayType>();
            shape.push_back(arrayType.getSize().getIntValue());
          }

          Type memrefElementType;
          if (arrayType.getElementType().isa<Float>()) {
            memrefElementType = FloatType::getF32(rewriter.getContext());
          } else if (arrayType.getElementType().isa<Int>()) {
            memrefElementType = IntegerType::get(32, rewriter.getContext());
          }

          auto array = rewriter.create<AllocOp>(
              loc, MemRefType::get(shape, memrefElementType));

          // For now just fill the array with one value
          StringRef litStr = literalValue;
          litStr = litStr.substr(0, litStr.find_first_of(','));
          litStr = litStr.trim('[');
          litStr = litStr.trim(']');
          float fillValue = std::stof(litStr.str() + ".0f");

          auto filler = rewriter.create<ConstantFloatOp>(
              loc, llvm::APFloat(fillValue),
              FloatType::getF32(rewriter.getContext()));

          rewriter.create<linalg::FillOp>(loc, array.getResult(),
                                          filler.getResult());
          //        std::cout << "\nHouston, we have a ArrayType Literal" <<
          //        std::flush;
          rewriter.restoreInsertionPoint(oldInsertPoint);
          return array.getResult();
        } else {
          emitError(op.getLoc())
              << "We can not lower literals of this type right now!";
        }
      }
    } else if (isa<LambdaOp>(contValue.getDefiningOp())) {
      emitRemark(contValue.getLoc()) << "ConT of Lambda";

      emitError(loc)
          << "We dont lower Lambdas using the function ConT right now.";
      // A Lambda has only one block
      Block &block = cast<LambdaOp>(contValue.getDefiningOp()).region().front();
      // For now start at the back and just find the first apply
      ApplyOp lastApply;
      for (auto op = block.rbegin(); op != block.rend(); op++) {
        if (isa<ApplyOp>(*op)) {
          lastApply = cast<ApplyOp>(*op);
          break;
        }
      }

      // Finding the return from the chunk of rise IR
      rise::ReturnOp returnOp = dyn_cast<rise::ReturnOp>(block.getTerminator());

    } else if (ApplyOp apply = dyn_cast<ApplyOp>(contValue.getDefiningOp())) {

      if (ZipOp zipOp = dyn_cast<ZipOp>(apply.fun().getDefiningOp())) {
        emitRemark(contValue.getLoc()) << "ConT of Applied Zip";

        auto lhs = apply.getOperand(1);
        auto rhs = apply.getOperand(2);

        auto contLhs = ConT(lhs, rewriter.getInsertionPoint(), rewriter);
        auto contRhs = ConT(rhs, rewriter.getInsertionPoint(), rewriter);

        // usually this is an Array of tuples. But at the end it always has to
        // be projected to fst or snd. For now I will keep the type as
        // memref<...xf32>
        MemRefType outputType =
            MemRefType::get({4}, FloatType::getF32(rewriter.getContext()));
        auto zipped = rewriter.create<RiseZipIntermediateOp>(
            zipOp.getLoc(), outputType, contLhs, contRhs);

        return zipped;
      } else if (FstOp fst = dyn_cast<FstOp>(apply.fun().getDefiningOp())) {
        emitRemark(contValue.getLoc()) << "ConT of Applied Fst";

        auto tuple = apply.getOperand(1);

        auto tupleCont = ConT(tuple, rewriter.getInsertionPoint(), rewriter);

        auto fstInterm = rewriter.create<RiseFstIntermediateOp>(
            fst.getLoc(), FloatType::getF32(rewriter.getContext()), tupleCont);
        return fstInterm;
      } else if (SndOp snd = dyn_cast<SndOp>(apply.fun().getDefiningOp())) {
        emitRemark(contValue.getLoc()) << "ConT of Applied Snd";

        auto tuple = apply.getOperand(1);

        auto tupleCont = ConT(tuple, rewriter.getInsertionPoint(), rewriter);

        auto sndInterm = rewriter.create<RiseSndIntermediateOp>(
            snd.getLoc(), FloatType::getF32(rewriter.getContext()), tupleCont);
        return sndInterm;
      } else if (MapSeqOp mapOp =
                     dyn_cast<MapSeqOp>(apply.fun().getDefiningOp())) {
        emitRemark(contValue.getLoc()) << "ConT of Applied Map";

        // introduce tmp Array of length n:
        auto tmpArray = rewriter.create<AllocOp>(
            loc, MemRefType::get(ArrayRef<int64_t>{mapOp.n().getIntValue()},
                                 FloatType::getF32(rewriter.getContext())));

        //      AccT(apply, tmpArray.getResult(), rewriter);
        //
        //      return tmpArray.getResult();
        AccT(apply, tmpArray.getResult(), rewriter);

        return tmpArray.getResult();
        // What do we really want to return here?

      } else if (MapParOp mapOp =
                     dyn_cast<MapParOp>(apply.fun().getDefiningOp())) {
        emitRemark(contValue.getLoc()) << "ConT of Applied Map";

        // introduce tmp Array of length n:
        auto tmpArray = rewriter.create<AllocOp>(
            loc, MemRefType::get(ArrayRef<int64_t>{mapOp.n().getIntValue()},
                                 FloatType::getF32(rewriter.getContext())));

        //      AccT(apply, tmpArray.getResult(), rewriter);
        //
        //      return tmpArray.getResult();
        AccT(apply, tmpArray.getResult(), rewriter);

        return tmpArray.getResult();
        // What do we really want to return here?

      } else {
        emitError(apply.getLoc()) << "Can not perform ConT for this apply!";
      }

    } else {
      emitRemark(contValue.getLoc())
          << "can not perform continuation "
             "translation for "
          << contValue.getDefiningOp()->getName().getStringRef().str()
          << " leaving Value as is.";

      rewriter.restoreInsertionPoint(oldInsertPoint);
      return contValue;
    }
  } else {
    emitRemark(contValue.getLoc())
        << "can not perform continuation for BlockArg, leaving as is.";
    return contValue;
  }

  //    std::cout << "\n 1\n" << op.literalAttr().getType().getKind() << " and
  //    "
  //    << RiseTypeKind::RISE_FLOAT;
  //
  //    if (op.literalAttr().getType().kindof(RiseTypeKind::RISE_ARRAY)) {
  //      std::cout << "\n 2 \n";
  //
  //    } else if (op.literalAttr().getValue()) {
  //      rewriter.create<RiseContinuationTranslation>(
  //          contValue.getLoc(), FloatType::getF32(rewriter.getContext()),
  //          contValue);
  //      std::cout << "\n 3\n";
  //
  //    }
  //  } else {
  //    emitError(contValue.getLoc())
  //        << "\nContinuation Translation of "
  //        << contValue.getDefiningOp()->getName().getStringRef().str()
  //        << " not "
  //           "supported.";
  //  }
}

Value mlir::rise::codeGen(Operation *op, SmallVector<OutputPathType, 10> path,
                          PatternRewriter &rewriter) {
  if (!op) {
    emitError(rewriter.getUnknownLoc()) << "codegen started with nullptr!";
  }
  if (RiseAssignOp assign = dyn_cast<RiseAssignOp>(op)) {
    emitRemark(op->getLoc()) << "Codegen for Assign";

    if (assign.value().isa<OpResult>()) {
      // Should be to the idx generated usually right beside this one
      // but I have no handle to that here
      rewriter.setInsertionPoint(assign.assignee().getDefiningOp());
    } else {
      rewriter.setInsertionPointToStart(
          &assign.value().getParentRegion()->front());
    }
    auto writeValue = codeGen(assign.value(), {}, rewriter);
    if (!writeValue)
      emitError(op->getLoc()) << "Assignment has no Value to write.";

    rewriter.setInsertionPointAfter(assign);
    auto leftPath = codeGenStore(assign.assignee(), writeValue, {}, rewriter);

  } else {
    emitRemark(op->getLoc())
        << "Codegen for " << op->getName().getStringRef().str()
        << " unsupported!";
  }
  return nullptr;
}
SmallVector<OutputPathType, 10>
mlir::rise::codeGenStore(Value storeLocation, Value val,
                         SmallVector<OutputPathType, 10> path,
                         PatternRewriter &rewriter) {
  if (storeLocation.isa<OpResult>()) {
    if (RiseIdxOp idx = dyn_cast<RiseIdxOp>(storeLocation.getDefiningOp())) {
      emitRemark(val.getLoc()) << "CodegenStore for idx";

      path.push_back(idx.arg1());

      //      std::cout <<
      //      idx.arg0().getDefiningOp()->getName().getStringRef().str()
      //                <<
      //                idx.arg1().getDefiningOp()->getName().getStringRef().str()
      //                <<
      //                storeLocation.getDefiningOp()->getName().getStringRef().str()
      //                << std::flush;
      return codeGenStore(idx.arg0(), val, path, rewriter);
    } else {
      emitRemark(val.getLoc())
          << "CodegenStore for "
          << val.getDefiningOp()->getName().getStringRef().str();

      generateReadAccess(path, val, storeLocation, rewriter);
    }
  } else {
    // We have reached a BlockArgument
    //    std::cout << "I have reached a BlockArg in codeGenStore\n";
    //    std::flush; print(path);
    // call to reverse here.
    //    auto index = path.pop_back_val();

    generateReadAccess(path, val, storeLocation, rewriter);
    //    rewriter.create<StoreOp>(val.getLoc(), val, storeLocation,
    //                             mpark::get<Value>(index));
  }
  // This might need an extra argument passing what Value exactly I want to
  // store.

  // First right hand side, get Value to store,
  // Then resolve lhs.

  // Look at the LateX requirements and create an Overleaf
  //
  return path;
}

Value mlir::rise::codeGen(Value val, SmallVector<OutputPathType, 10> path,
                          PatternRewriter &rewriter) {

  if (val.isa<OpResult>()) {
    if (RiseIdxOp idx = dyn_cast<RiseIdxOp>(val.getDefiningOp())) {
      emitRemark(idx.getLoc()) << "Codegen for idx";

      Value arg1 = idx.arg1();
      path.push_back(arg1);
      //      rewriter.setInsertionPointAfter(idx);
      return codeGen(idx.arg0(), path, rewriter);

    } else if (RiseBinaryOp binOp =
                   dyn_cast<RiseBinaryOp>(val.getDefiningOp())) {
      emitRemark(binOp.getLoc()) << "Codegen for binOp";

      //      std::cout << "\nCodeGen for binOp!" << std::flush;
      auto arg0 = codeGen(binOp.arg0(), {}, rewriter);
      auto arg1 = codeGen(binOp.arg1(), {}, rewriter);
      if (binOp.op().equals("add")) {
        return rewriter.create<AddFOp>(val.getLoc(), arg0.getType(), arg0, arg1)
            .getResult();
      } else if (binOp.op().equals("mul")) {
        return rewriter.create<MulFOp>(val.getLoc(), arg0.getType(), arg0, arg1)
            .getResult();
      } else {
        emitError(binOp.getLoc())
            << "Cannot create code for binOp:" << binOp.op();
        return binOp.getResult();
      }
    } else if (AllocOp alloc = dyn_cast<AllocOp>(val.getDefiningOp())) {
      emitRemark(alloc.getLoc()) << "Codegen for alloc";

      // call to reverse here.
      return generateWriteAccess(path, alloc.getResult(), rewriter);

    } else if (RiseZipIntermediateOp zipIntermOp =
                   dyn_cast<RiseZipIntermediateOp>(val.getDefiningOp())) {
      emitRemark(zipIntermOp.getLoc()) << "Codegen for zip";

      //      std::cout << "\nCodeGen for zipInterm!" << std::flush;

      // now there has to be an idx on the path.
      //      printPath(path);
      //      auto topPath = path.pop_back_val();

      //      auto sndPath = path.pop_back_val();
      //      OutputPathType topElement = path.pop_back_val();
      //      Value *idx = mpark::get_if<Value>(&topElement);
      //      std::cout << "arg in zip: " << idx->getType().isa<IndexType>()
      //                << std::flush;

      //      printPath(path);
      OutputPathType sndLastElem = path[path.size() - 2];
      int *fst = mpark::get_if<int>(&sndLastElem);

      // delete snd value on the path.
      auto tmp = path.pop_back_val();
      path.pop_back();
      path.push_back(tmp);

      //      std::cout << "fst? : " << *fst << "\n" << std::flush;
      if (*fst) {
        return codeGen(zipIntermOp.lhs(), path, rewriter);
      } else {
        return codeGen(zipIntermOp.rhs(), path, rewriter);
      }
    } else if (RiseFstIntermediateOp fstIntermOp =
                   dyn_cast<RiseFstIntermediateOp>(val.getDefiningOp())) {
      emitRemark(fstIntermOp.getLoc()) << "Codegen for fst";

      path.push_back(true);
      return codeGen(fstIntermOp.value(), path, rewriter);

    } else if (RiseSndIntermediateOp sndIntermOp =
                   dyn_cast<RiseSndIntermediateOp>(val.getDefiningOp())) {
      emitRemark(sndIntermOp.getLoc()) << "Codegen for snd";

      path.push_back(false);
      return codeGen(sndIntermOp.value(), path, rewriter);

    } else if (isa<LoadOp>(val.getDefiningOp()) ||
               isa<AffineLoadOp>(val.getDefiningOp())) {
      emitRemark(val.getLoc()) << "Codegen for Load";
      return val;
    } else {
      emitRemark(val.getLoc())
          << "I don't know how to do codegen for: "
          << val.getDefiningOp()->getName().getStringRef().str()
          << " this is prob. an operation from another dialect. We walk "
             "recursively through the operands until we hit something we can "
             "do codegen for.";
      int i = 0;
      for (auto operand : (val.getDefiningOp()->getOperands())) {
        val.getDefiningOp()->setOperand(i, codeGen(operand, path, rewriter));
        i++;
      }
      return val;
      // go through all the operands until we hit an idx
    }
  } else {
    // val is a BlockArg
    emitRemark(val.getLoc()) << "reached a blockArg in Codegen, reversing";
    return generateWriteAccess(path, val, rewriter);
  }

  emitError(val.getLoc())
      << "Something went wrong in codegen, we should never get here!";
  return val;
}

Value mlir::rise::generateWriteAccess(SmallVector<OutputPathType, 10> path,
                                      Value accessVal,
                                      PatternRewriter &rewriter) {
  //  std::cout << "reversing Path for writing.\n" << std::flush;
  //  printPath(path);
  int index;
  SmallVector<Value, 10> indexValues = {};
  for (OutputPathType element : path) {
    if (auto i = mpark::get_if<int>(&element)) {
      index = *i;
    } else if (auto val = mpark::get_if<Value>(&element)) {
      indexValues.push_back(*val);
    }
  }
  // handle problem originating from translation of reduce (accessing element)
  int rank = accessVal.getType().dyn_cast<MemRefType>().getRank();
  if (indexValues.size() != rank) {
    indexValues.erase(indexValues.begin());
  }
  if (isa<AffineForOp>(rewriter.getBlock()->getParent()->getParentOp())) {
    return rewriter
        .create<AffineLoadOp>(accessVal.getLoc(), accessVal, indexValues)
        .getResult();
  } else {
    return rewriter.create<LoadOp>(accessVal.getLoc(), accessVal, indexValues)
        .getResult();
  }
}

void mlir::rise::generateReadAccess(SmallVector<OutputPathType, 10> path,
                                    Value storeVal, Value storeLoc,
                                    PatternRewriter &rewriter) {
  //  std::cout << "reversing Path for reading\n" << std::flush;
  //  printPath(path);

  int index;
  SmallVector<Value, 10> indexValues = {};
  for (OutputPathType element : path) {
    if (auto i = mpark::get_if<int>(&element)) {
      index = *i;
    } else if (auto val = mpark::get_if<Value>(&element)) {
      indexValues.push_back(*val);
    }
  }
  int rank = storeLoc.getType().dyn_cast<MemRefType>().getRank();
  if (indexValues.size() != rank) {
    indexValues.erase(indexValues.begin());
  }
  ValueRange valRange = ValueRange(indexValues);
  if (isa<AffineForOp>(rewriter.getBlock()->getParent()->getParentOp())) {
    rewriter.create<AffineStoreOp>(storeLoc.getLoc(), storeVal, storeLoc,
                                   llvm::makeArrayRef(indexValues));
    return;
  } else {
    rewriter.create<StoreOp>(storeLoc.getLoc(), storeVal, storeLoc,
                             llvm::makeArrayRef(indexValues));

    return;
  }
}

/// This is obviously not really working.
/// For some Values it prints ints.
void mlir::rise::printPath(SmallVector<OutputPathType, 10> input) {
  struct {
    void operator()(Value val) {
      if (val.isa<OpResult>()) {
        std::cout << "val: "
                  << val.getDefiningOp()->getName().getStringRef().str() << " ";
      } else {
        std::cout << "blockArg, ";
      }
    }
    void operator()(Value *val) {
      if (val->isa<OpResult>()) {
        std::cout << "val: "
                  << val->getDefiningOp()->getName().getStringRef().str()
                  << " ";
      } else {
        std::cout << "blockArg, ";
      }
    }
    void operator()(int i) { std::cout << "int!" << i << ", "; }
    void operator()(std::string const &) { std::cout << "string!"; }
    void operator()(bool b) { std::cout << "bool: " << b << ", "; }
  } visitor;
  std::cout << "path: {";
  for (OutputPathType element : input) {
    mpark::visit(visitor, input[0]);
  }
  std::cout << "}\n" << std::flush;
}

void mlir::rise::printUses(Value val) {

  //  auto uses = val.getUses();
  //  auto iter = uses.begin();
  //  std::cout << "first use of"
  //            << val.getDefiningOp()->getName().getStringRef().str() << "
  //            is:
  //            "
  //            <<
  //            val.getUses().begin().getUser()->getName().getStringRef().str()
  //            << " with id "
  //            << "\n"
  //            << std::flush;
  std::cout << val.getDefiningOp()->getName().getStringRef().str()
            << " has uses: \n"
            << std::flush;

  auto uses = val.getUses().begin();
  while (true) {
    if (uses != val.getUses().end()) {
      std::cout << "    " << uses.getUser()->getName().getStringRef().str()
                << "\n"
                << std::flush;
      uses++;
    } else {
      break;
    }
  }
}

void mlir::rise::Substitute(LambdaOp lambda,
                            llvm::SmallVector<Value, 10> args) {
  if (lambda.region().front().getArguments().size() < args.size()) {
    emitError(lambda.getLoc())
        << "Too many arguments given for Lambda substitution";
  }
  // TODO: right now we can not do Substitution more than once for a single
  // lambda. We want to be able to do that to use a Lambda more than once for
  // translation.
  for (int i = 0; i < args.size(); i++) {
    lambda.region().front().getArgument(i).replaceAllUsesWith(args[i]);
    //    lambda.region().front().insertArgument(i, args[i].getType());
  }
  // TODO: look at how the IR looks at this point to find out how we can do
  // Substitution more than once.
  return;
}

/// gather all patterns
void mlir::populateRiseToImpConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<RiseToImperativePattern>(ctx);
}
//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// The pass:
void ConvertRiseToImperativePass::runOnFunction() {
  auto module = getOperation();
  OwningRewritePatternList patterns;

  populateRiseToImpConversionPatterns(patterns, &getContext());

  ConversionTarget target(getContext());

  //  target.addLegalOp<
  //      CallOp, FuncOp, ModuleOp, ModuleTerminatorOp, linalg::MatmulOp,
  //      scf::ForOp, scf::ParallelOp, scf::TerminatorOp, AffineForOp,
  //      AffineParallelOp, AffineTerminatorOp, AffineStoreOp, AffineLoadOp,
  //      ConstantIndexOp, AllocOp, LoadOp, StoreOp, AddFOp, MulFOp,
  //      linalg::FillOp, mlir::ReturnOp, mlir::rise::LambdaOp,
  //      mlir::rise::RiseIdxOp, mlir::rise::RiseBinaryOp,
  //      mlir::rise::RiseFstIntermediateOp, mlir::rise::RiseSndIntermediateOp,
  //      mlir::rise::RiseZipIntermediateOp, mlir::rise::RiseAssignOp,
  //      mlir::rise::RiseWrapOp, mlir::rise::RiseUnwrapOp, mlir::rise::ApplyOp,
  //      RiseContinuationTranslation>();

  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addLegalDialect<rise::RiseDialect>(); // for debugging purposes

  // Ops we don't want in our output
  target.addDynamicallyLegalOp<FuncOp>([](FuncOp funcOp) {
    bool riseInside = false;
    if (funcOp.isExternal())
      return true;
    funcOp.walk([&](Operation *op) {
      if (op->getDialect()->getNamespace().equals(
              rise::RiseDialect::getDialectNamespace()))
        riseInside = true;
    });
    return !riseInside;
  });

  //  if (failed(applyPartialConversion(module, target, patterns)))
  //    signalPassFailure();

  //        if (!applyPatternsGreedily(this->getOperation(), patterns))

  //  std::cout << "here I am! \n" << std::flush;

  bool erased;
  applyOpPatternsAndFold(module, patterns, &erased);

  //  std::cout << "here I am! erased:" << erased << "\n" << std::flush;

  return;
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::rise::createConvertRiseToImperativePass() {
  return std::make_unique<ConvertRiseToImperativePass>();
}
