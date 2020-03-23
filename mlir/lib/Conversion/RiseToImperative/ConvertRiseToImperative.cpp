//
// Created by martin on 26/11/2019.
//

#include "mlir/Conversion/RiseToImperative/ConvertRiseToImperative.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/EDSC/Builders.h>

using namespace mlir;
using namespace mlir::rise;

namespace {
struct ConvertRiseToImperativePass
    : public ModulePass<ConvertRiseToImperativePass> {
  void runOnModule() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//
/// Apply
struct ApplyToImpLowering : public OpRewritePattern<ApplyOp> {
  using OpRewritePattern<ApplyOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ApplyOp applyOp,
                                     PatternRewriter &rewriter) const override;
};

PatternMatchResult
ApplyToImpLowering::matchAndRewrite(ApplyOp applyOp,
                                    PatternRewriter &rewriter) const {
  MLIRContext *context = rewriter.getContext();
  Location loc = applyOp.getLoc();
  // TODO: do
  //    emitError(loc) << "yoyoyo";
  return matchSuccess();
}

struct ModuleToImp : public OpRewritePattern<RiseFunOp> {
  using OpRewritePattern<RiseFunOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(RiseFunOp riseFunOp,
                                     PatternRewriter &rewriter) const override;
};

PatternMatchResult
ModuleToImp::matchAndRewrite(RiseFunOp riseFunOp,
                             PatternRewriter &rewriter) const {
  MLIRContext *context = rewriter.getContext();
  Location loc = riseFunOp.getLoc();
  Region &riseRegion = riseFunOp.region();

  if (!riseRegion.getParentRegion()->getParentRegion()) {
    emitError(loc) << "Rise IR cant be immediately nested into a module. It "
                      "has to be surrounded by e.g. a FuncOp.";
    return matchFailure();
  }

  // create mlir function for the given chunk of rise and inline all rise
  // operations
  rewriter.setInsertionPointToStart(
      &riseRegion.getParentRegion()->getParentRegion()->front());

  // find declaration for the riseFun and delete
  auto forwardDeclaration =
      riseRegion.getParentOfType<ModuleOp>().lookupSymbol(riseFunOp.name());
  rewriter.eraseOp(forwardDeclaration);

  // The in is tmp for now.
  auto riseFun = rewriter.create<FuncOp>(
      loc, riseFunOp.name(),
      FunctionType::get(riseFunOp.region().front().getArgument(0).getType(), {},
                        context),
      ArrayRef<NamedAttribute>{});

  rewriter.setInsertionPointToStart(&riseFunOp.getParentRegion()->front());
  //  auto callRiseFunOp = rewriter.create<CallOp>(riseFun.getLoc(), riseFun);
  riseFun.addEntryBlock();

  // The rise module can only have one block.
  Block &block = riseRegion.front();

  // For now there can be only one output
  RiseOutOp outOp;
  for (auto op = block.begin(); op != block.end(); op++) {
    if (isa<RiseOutOp>(*op)) {
      outOp = cast<RiseOutOp>(*op);
      break;
    }
  }

  // For now start at the back and just find the first apply.
  ApplyOp lastApply;
  for (auto op = block.rbegin(); op != block.rend(); op++) {
    if (isa<ApplyOp>(*op)) {
      lastApply = cast<ApplyOp>(*op);
      break;
    }
  }
  // Finding the return from the chunk of rise IR
  rise::ReturnOp returnOp;
  for (auto op = block.rbegin(); op != block.rend(); op++) {
    if (isa<rise::ReturnOp>(*op)) {
      returnOp = cast<rise::ReturnOp>(*op);
      break;
    }
  }

  // Translation to imperative
  rewriter.setInsertionPointToStart(&riseFun.getBody().front());
  auto result =
      AccT(lastApply, riseFunOp.region().front().getArgument(0), rewriter);

  emitRemark(riseFun.getLoc()) << "AccT finished. Starting CodeGen.";

  // codegen here
  //  std::cout << "\nmain AccT finished!" << std::flush;

  // For now just find the assign ad-hoc and go from there. Later think about
  // where to start with Codegen
  rise::RiseAssignOp assign;
  riseFun.walk([&assign](Operation *inst) {
    if (isa<RiseAssignOp>(inst)) {
      assign = cast<RiseAssignOp>(inst);
    }
    return;
  });
  for (auto opFor = riseFun.getBody().front().rbegin();
       opFor != riseFun.getBody().front().rend(); opFor++) {
    if (isa<mlir::loop::ForOp>(*opFor)) {
      for (auto op = cast<loop::ForOp>(*opFor).getBody()->rbegin();
           op != cast<loop::ForOp>(*opFor).getBody()->rend(); op++) {
        if (isa<rise::RiseAssignOp>(*op)) {
          assign = cast<rise::RiseAssignOp>(*op);
          break;
        }
      }
    }
  }
//
  codeGen(assign, {}, rewriter);
//
  // erase intermediate operations. We remove them back to front right now,
  // this should be alright in terms of dependencies.
  SmallVector<Operation *, 10> erasureList = {};
  riseFun.walk([&erasureList](Operation *inst) {
    if (isa<RiseAssignOp>(inst) || isa<RiseBinaryOp>(inst) ||
        isa<RiseIdxOp>(inst) || isa<RiseZipIntermediateOp>(inst) ||
        isa<RiseFstIntermediateOp>(inst) || isa<RiseSndIntermediateOp>(inst)) {
      erasureList.push_back(inst);
    }
    return;
  });
  // cleanup

  size_t unneededOps = erasureList.size();
  for (size_t i = 0; i < unneededOps; i++) {
    auto op = erasureList.pop_back_val();
    op->dropAllUses();
    op->dropAllReferences();
    rewriter.eraseOp(op);
  }

  rewriter.setInsertionPointToEnd(&riseFun.getBody().back());
  auto newReturn =
      rewriter.create<mlir::ReturnOp>(returnOp.getLoc()); //,result);

  // We don't need the riseModule anymore
  // replace output
  riseFunOp.region().front().getArgument(0).replaceAllUsesWith(
      riseFun.getBody().front().getArgument(0));

  riseFunOp.walk([&rewriter](Operation *inst) {
    if (!inst->use_empty()) {
      //      std::cout << "printing uses for " <<
      //      inst->getName().getStringRef().str()
      //                << "\n"
      //                << std::flush;

      //      inst->dropAllDefinedValueUses();
      inst->getResult(0).dropAllUses();
      //        printUses(inst->getResult(0));

      //      rewriter.eraseOp(inst);
    } else {
      //        std::cout << "walking over: " <<
      //        inst->getName().getStringRef().str()
      //                  << "\n"
      //                  << std::flush;
    }
    return;
  });

  riseFunOp.getOperation()->dropAllUses();
  riseFunOp.getOperation()->dropAllReferences();
  rewriter.eraseOp(riseFunOp);
  return matchSuccess();
}
// std::cout << "\n" << "" << "\n" << std::flush;

// TODO: look at Map(Map(id))

/// Acceptor Translation
/// apply - The ApplyOp to start the translation.
/// outsideArgs - any mlir::Value which are operands of rise.fun or a lambda
mlir::Value mlir::rise::AccT(ApplyOp apply, Value out,
                             PatternRewriter &rewriter) {
  // lower outsideArgs first

  Operation *appliedFun = apply.getOperand(0).getDefiningOp();

  // If functions are applied partially i.e. appliedFun is an ApplyOp we iterate
  // here until we reach a non ApplyOp
  // We push the operands of the applies into a vector, as only the effectively
  // applied Op knows what to do with them.
  // We always leave out operand 0, as this is the applied function
  // Operands in the Vector will be ordered such that the top most operand is
  // the first needed to the applied function. As applies are processed
  // bottom-up, in the case that an apply has more than 1 operand they have to
  // be added right to left to keep order.
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
  if (isa<ReduceOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Reduce";

    auto n = appliedFun->getAttrOfType<NatAttr>("n");
    auto s = appliedFun->getAttrOfType<DataTypeAttr>("s");
    auto t = appliedFun->getAttrOfType<DataTypeAttr>("t");

    auto reductionFun = applyOperands.pop_back_val();

    auto initializer = applyOperands.pop_back_val();
    auto array = applyOperands.pop_back_val();

    // Add Continuation for array.
    auto contArray = ConT(array, rewriter.getInsertionPoint(), rewriter);

    // Add Continuation for init
    auto contInit = ConT(initializer, rewriter.getInsertionPoint(), rewriter);

    // Accumulator for Reduction
    auto acc = rewriter.create<AllocOp>(
        appliedFun->getLoc(),
        MemRefType::get(ArrayRef<int64_t>{1},
                        FloatType::getF32(rewriter.getContext())));

    rewriter.create<linalg::FillOp>(initializer.getLoc(), acc.getResult(),
                                    contInit);

    auto lowerBound = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 0);
    auto upperBound = rewriter.create<ConstantIndexOp>(
        appliedFun->getLoc(), n.getValue().getIntValue());
    auto step = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 1);

    auto forLoop =
        rewriter.create<mlir::loop::ForOp>(loc, lowerBound, upperBound, step);

    // create operations for addition inside the loop
    rewriter.setInsertionPointToStart(forLoop.getBody());
//    auto x1 = rewriter.create<LoadOp>(reductionFun.getLoc(), acc.getResult(),
//                                      lowerBound.getResult());
//    auto x2 = rewriter.create<LoadOp>(reductionFun.getLoc(), contArray,
//                                      forLoop.getInductionVar());

    LambdaOp reductionLambda = dyn_cast<LambdaOp>(reductionFun.getDefiningOp());
    if (!reductionLambda) {
      reductionLambda = expandToLambda(reductionFun, rewriter);
    }

    RiseIdxOp xi;

    if (contArray.getType().isa<MemRefType>()) {
      ArrayRef<int64_t> inShape =
          contArray.getType().dyn_cast<MemRefType>().getShape().drop_back(1);

      MemRefType inIndexOpResult =
          MemRefType::get(inShape, FloatType::getF32(rewriter.getContext()));

      xi = rewriter.create<RiseIdxOp>(loc, inIndexOpResult, contArray,
                                      forLoop.getInductionVar());
    } else if (isa<RiseIdxOp>(contArray.getDefiningOp())) {
    }

    // operate on a copy of the lambda to avoid generating dependencies.
    LambdaOp lambdaCopy = cast<LambdaOp>(rewriter.clone(*reductionLambda));
    auto fxi = rewriter.create<ApplyOp>(loc, lambdaCopy.getType(),
                                        lambdaCopy.getResult(), ValueRange{xi.getResult(), acc.getResult()});

    // This generated idx is not consistent with the bible
    // I just generate a 0 as index for the simplest case.
    ArrayRef<int64_t> outShape =
        out.getType().dyn_cast<MemRefType>().getShape().drop_back(1);
    MemRefType outIndexOpResult =
        MemRefType::get(outShape, FloatType::getF32(rewriter.getContext()));
    auto outi = rewriter.create<RiseIdxOp>(loc, outIndexOpResult, out,
                                           lowerBound.getResult());

    AccT(fxi, outi.getResult(), rewriter);

    fxi.getResult().dropAllUses();
    fxi.erase();

    lambdaCopy.getResult().dropAllUses();
    rewriter.eraseOp(lambdaCopy);



//    Substitute(reductionLambda, {x1.getResult(), x2.getResult()});
//
//    rewriter.setInsertionPointAfter(x2);
//    auto lambdaResult = AccT(lastApply, {}, rewriter);
//    rewriter.setInsertionPointAfter(lambdaResult.getDefiningOp());
//
//    auto storing =
//        rewriter.create<StoreOp>(reductionFun.getLoc(), lambdaResult,
//                                 acc.getResult(), lowerBound.getResult());
//
    return acc.getResult();

  } else if (isa<MapOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Map";

    //     For now we treat all maps as mapSeqs
    auto n = appliedFun->getAttrOfType<NatAttr>("n");
    auto s = appliedFun->getAttrOfType<DataTypeAttr>("s");
    auto t = appliedFun->getAttrOfType<DataTypeAttr>("t");

    auto f = applyOperands.pop_back_val();

    auto array = applyOperands.pop_back_val();

    auto contArray = ConT(array, rewriter.getInsertionPoint(), rewriter);

    auto lowerBound = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 0);
    auto upperBound = rewriter.create<ConstantIndexOp>(
        appliedFun->getLoc(), n.getValue().getIntValue());
    auto step = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 1);

    auto forLoop =
        rewriter.create<mlir::loop::ForOp>(loc, lowerBound, upperBound, step);

    rewriter.setInsertionPointToStart(forLoop.getBody());

    LambdaOp fLambda = dyn_cast<LambdaOp>(f.getDefiningOp());
    if (!fLambda) {
      fLambda = expandToLambda(f, rewriter);
    }

    // TODO: create a custom builder (or create?) for RiseIdxOp and call this
    // one here, so I dont have to give the resulting type.
    //    std::cout << "still here! \n" << std::flush;

    // We could also already have an idx here.
    // put this into its own functiono

    RiseIdxOp xi;

    if (contArray.getType().isa<MemRefType>()) {
      //      std::cout << "got a memref! Shape: \n"
      //                <<
      //                contArray.getType().dyn_cast<MemRefType>().getShape().size()
      //                << ","
      //                <<
      //                contArray.getType().dyn_cast<MemRefType>().getShape()[0]
      //                << std::flush;

      ArrayRef<int64_t> inShape =
          contArray.getType().dyn_cast<MemRefType>().getShape().drop_back(1);

      MemRefType inIndexOpResult =
          MemRefType::get(inShape, FloatType::getF32(rewriter.getContext()));

      xi = rewriter.create<RiseIdxOp>(loc, inIndexOpResult, contArray,
                                      forLoop.getInductionVar());

      //      std::cout << "created idxOp with args: " << isa<Value>(xi.arg0())
      //                << isa<Value>(xi.arg1()) << "\n"
      //                << std::flush;
      //      std::cout << "arg1: " << xi.arg1().getType().isa<IndexType>()
      //                << std::flush;
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
                                           forLoop.getInductionVar());

    AccT(fxi, outi.getResult(), rewriter);
    // tmp Apply not needed anymore.

    //    std::cout << "deleting apply dependency for lambda" << std::flush;
    fxi.getResult().dropAllUses();
    fxi.erase();
    //    rewriter.eraseOp(fxi);

    lambdaCopy.getResult().dropAllUses();
    rewriter.eraseOp(lambdaCopy);

    // create Apply for Lambda now, and create other idx for the out.
    // start AccT for Apply of Lambda and add case for Lamda

    //    // This LoadOp should not be created here.
    //        auto x1 = rewriter.create<LoadOp>(appliedFun->getLoc(), contArray,
    //                                          forLoop.getInductionVar());

    // Lower mapped function
    //    fLambda.region().front().getArgument(0).replaceAllUsesWith(x1.getResult());
    // search for the last apply inside the lambda and start lowering from there
    //    ApplyOp lastApply;
    //    for (auto op = fLambda.region().front().rbegin();
    //         op != fLambda.region().front().rend(); op++) {
    //      if (isa<ApplyOp>(*op)) {
    //        lastApply = cast<ApplyOp>(*op);
    //        break;
    //      }
    //    }
    // instead have a case for Lambda.
    // generate an Apply for the Lambda

    //    Substitute(fLambda, {x1.getResult()});

    // What would the argument for the storing look like?
    // I would pass the outputArr and the "Path"? -> what exactly is the "o"?
    //    auto lambdaResult = AccT(lastApply, {}, rewriter);

    //    rewriter.setInsertionPointAfter(lambdaResult.getDefiningOp());

    //    auto storing = rewriter.create<StoreOp>(appliedFun->getLoc(),
    //    lambdaResult,
    //                                            out.getDefiningOp()->getOperand(0),
    //                                            forLoop.getInductionVar());
    return contArray;

  } else if (FstOp fstOp = dyn_cast<FstOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Fst";

    auto tuple = applyOperands.pop_back_val();
    auto contTuple = ConT(tuple, rewriter.getInsertionPoint(), rewriter);

    auto fstIntermOp = rewriter.create<RiseFstIntermediateOp>(
        fstOp.getLoc(), FloatType::getF32(rewriter.getContext()), contTuple);
    auto assignment = rewriter.create<RiseAssignOp>(
        appliedFun->getLoc(), fstIntermOp.getResult(), out);
    return fstIntermOp.getResult();

  } else if (SndOp sndOp = dyn_cast<SndOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Snd";

    auto tuple = applyOperands.pop_back_val();
    auto contTuple = ConT(tuple, rewriter.getInsertionPoint(), rewriter);

    auto sndIntermOp = rewriter.create<RiseSndIntermediateOp>(
        sndOp.getLoc(), FloatType::getF32(rewriter.getContext()), contTuple);
    auto assignment = rewriter.create<RiseAssignOp>(
        appliedFun->getLoc(), sndIntermOp.getResult(), out);
    return sndIntermOp.getResult();

  } else if (isa<LambdaOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Lambda";

    LambdaOp lambda = cast<LambdaOp>(appliedFun);
    Substitute(lambda, applyOperands);

    // Find last Apply inside Lambda:
    ApplyOp lastApply;
    for (auto op = lambda.region().front().rbegin();
         op != lambda.region().front().rend(); op++) {
      if (isa<ApplyOp>(*op)) {
        lastApply = cast<ApplyOp>(*op);
        break;
      }
    }

    return AccT(lastApply, out, rewriter);

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
        contSummand0, contSummand1);
    auto assignment = rewriter.create<RiseAssignOp>(appliedFun->getLoc(),
                                                    newAddOp.getResult(), out);

    return newAddOp;

  } else if (isa<MultOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Mult";

    auto factor0 = applyOperands.pop_back_val();
    auto factor1 = applyOperands.pop_back_val();

    auto contFactor0 = ConT(factor0, rewriter.getInsertionPoint(), rewriter);
    auto contFactor1 = ConT(factor1, rewriter.getInsertionPoint(), rewriter);

    auto newMultOp = rewriter.create<RiseBinaryOp>(
        appliedFun->getLoc(), FloatType::getF32(rewriter.getContext()),
        contFactor0, contFactor1);

    auto assignment = rewriter.create<RiseAssignOp>(appliedFun->getLoc(),
                                                    newMultOp.getResult(), out);

    return newMultOp;
  } else if (isa<ApplyOp>(appliedFun)) {
    emitRemark(appliedFun->getLoc()) << "AccT of Apply";

    emitError(appliedFun->getLoc()) << "We should never get here";
    //    auto tmp = AccT(expression, cast<ApplyOp>(appliedFun), rewriter);

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

  if (isa<LiteralOp>(contValue.getDefiningOp())) {
    emitRemark(contValue.getLoc()) << "ConT of Literal";

    StringRef literalValue =
        dyn_cast<LiteralOp>(contValue.getDefiningOp()).literalAttr().getValue();

    //    //  TODO: I should be doing this. But this does not work
    //        std::cout << "\nTest printing";
    //        if (LiteralOp op = dyn_cast<LiteralOp>(contValue.getDefiningOp()))
    //        {
    //          if (op.literalAttr().getType().kindof(RiseTypeKind::RISE_FLOAT))
    //          {
    //            std::cout << "\nHouston, we have a Float Literal" <<
    //            std::flush;
    //          } else {
    //            std::cout << "\nHouston, we dont have a Float Literal" <<
    //            std::flush;
    //          }
    //          //  or this:
    //          if (op.literalAttr().getType().isa<mlir::rise::Float>()) {
    //            std::cout << "\nHouston, we have a Float Literal" <<
    //            std::flush;
    //          } else {
    //            std::cout << "\nHouston, we dont have a Float Literal" <<
    //            std::flush;
    //          }
    //          // or:
    //          if (Float num =
    //          op.literalAttr().getType().dyn_cast<mlir::rise::Float>()) {
    //            std::cout << "\nHouston, we have a Float Literal" <<
    //            std::flush;
    //          } else {
    //            std::cout << "\nHouston, we dont have a Float Literal" <<
    //            std::flush;
    //          }
    //        }

    // This should of course check for an int or float type
    // However the casting for some reason does not work. I'll hardcode it for
    // now
    if (literalValue == "0") {
      auto fillOp = rewriter.create<ConstantFloatOp>(
          loc, llvm::APFloat(0.0f), FloatType::getF32(rewriter.getContext()));

      rewriter.restoreInsertionPoint(oldInsertPoint);
      return fillOp.getResult();
    } else if (literalValue == "5") {
      auto fillOp = rewriter.create<ConstantFloatOp>(
          loc, llvm::APFloat(5.0f), FloatType::getF32(rewriter.getContext()));

      rewriter.restoreInsertionPoint(oldInsertPoint);
      return fillOp.getResult();
    }
    // This should check for an array type
    else if (literalValue == "[5,5,5,5]") {
      auto array = rewriter.create<AllocOp>(
          loc, MemRefType::get(ArrayRef<int64_t>{4},
                               FloatType::getF32(rewriter.getContext())));
      // For now just fill the array with one value
      auto filler = rewriter.create<ConstantFloatOp>(
          loc, llvm::APFloat(5.0f), FloatType::getF32(rewriter.getContext()));

      rewriter.create<linalg::FillOp>(loc, array.getResult(),
                                      filler.getResult());

      rewriter.restoreInsertionPoint(oldInsertPoint);
      return array.getResult();

      //    return rewriter.create<RiseContinuationTranslation>(
      //        contValue.getLoc(),
      //        MemRefType::get(ArrayRef<int64_t>{32},
      //                        FloatType::getF32(rewriter.getContext())),
      //        contValue);
    } else if (literalValue == "[10,10,10,10]") {
      auto array = rewriter.create<AllocOp>(
          loc, MemRefType::get(ArrayRef<int64_t>{4},
                               FloatType::getF32(rewriter.getContext())));
      // For now just fill the array with one value
      auto filler = rewriter.create<ConstantFloatOp>(
          loc, llvm::APFloat(10.0f), FloatType::getF32(rewriter.getContext()));

      rewriter.create<linalg::FillOp>(loc, array.getResult(),
                                      filler.getResult());

      rewriter.restoreInsertionPoint(oldInsertPoint);
      return array.getResult();

      //    return rewriter.create<RiseContinuationTranslation>(
      //        contValue.getLoc(),
      //        MemRefType::get(ArrayRef<int64_t>{32},
      //                        FloatType::getF32(rewriter.getContext())),
      //        contValue);
    } else if (literalValue == "[[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]]") {
      auto array = rewriter.create<AllocOp>(
          loc, MemRefType::get(ArrayRef<int64_t>{4, 4},
                               FloatType::getF32(rewriter.getContext())));
      // For now just fill the array with one value
      auto filler = rewriter.create<ConstantFloatOp>(
          loc, llvm::APFloat(5.0f), FloatType::getF32(rewriter.getContext()));

      rewriter.create<linalg::FillOp>(loc, array.getResult(),
                                      filler.getResult());

      rewriter.restoreInsertionPoint(oldInsertPoint);
      return array.getResult();
    } else if (literalValue ==
               "[[[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]], [[5,5,5,5], "
               "[5,5,5,5], [5,5,5,5], [5,5,5,5]], [[5,5,5,5], [5,5,5,5], "
               "[5,5,5,5], [5,5,5,5]], [[5,5,5,5], [5,5,5,5], [5,5,5,5], "
               "[5,5,5,5]]]") {
      auto array = rewriter.create<AllocOp>(
          loc, MemRefType::get(ArrayRef<int64_t>{4, 4, 4},
                               FloatType::getF32(rewriter.getContext())));
      // For now just fill the array with one value
      auto filler = rewriter.create<ConstantFloatOp>(
          loc, llvm::APFloat(5.0f), FloatType::getF32(rewriter.getContext()));

      rewriter.create<linalg::FillOp>(loc, array.getResult(),
                                      filler.getResult());

      rewriter.restoreInsertionPoint(oldInsertPoint);
      return array.getResult();
    } else if (literalValue ==
               "[[[[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]], [[5,5,5,5], "
               "[5,5,5,5], [5,5,5,5], [5,5,5,5]], [[5,5,5,5], [5,5,5,5], "
               "[5,5,5,5], [5,5,5,5]], [[5,5,5,5], [5,5,5,5], [5,5,5,5], "
               "[5,5,5,5]]], [[[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]], "
               "[[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]], [[5,5,5,5], "
               "[5,5,5,5], [5,5,5,5], [5,5,5,5]], [[5,5,5,5], [5,5,5,5], "
               "[5,5,5,5], [5,5,5,5]]], [[[5,5,5,5], [5,5,5,5], [5,5,5,5], "
               "[5,5,5,5]], [[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]], "
               "[[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]], [[5,5,5,5], "
               "[5,5,5,5], [5,5,5,5], [5,5,5,5]]], [[[5,5,5,5], [5,5,5,5], "
               "[5,5,5,5], [5,5,5,5]], [[5,5,5,5], [5,5,5,5], [5,5,5,5], "
               "[5,5,5,5]], [[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]], "
               "[[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]]]]") {
      auto array = rewriter.create<AllocOp>(
          loc, MemRefType::get(ArrayRef<int64_t>{4, 4, 4, 4},
                               FloatType::getF32(rewriter.getContext())));
      // For now just fill the array with one value
      auto filler = rewriter.create<ConstantFloatOp>(
          loc, llvm::APFloat(5.0f), FloatType::getF32(rewriter.getContext()));

      rewriter.create<linalg::FillOp>(loc, array.getResult(),
                                      filler.getResult());

      rewriter.restoreInsertionPoint(oldInsertPoint);
      return array.getResult();
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

      // usually this is an Array of tuples. But at the end it always has to be
      // projected to fst or snd. For now I will keep the type as
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


    auto writeValue = codeGen(assign.value(), {}, rewriter);
    if (!writeValue)
      emitError(op->getLoc()) << "Assignment has no Value to write to.";
    auto leftPath = codeGenStore(assign.assignee(), writeValue, {}, rewriter);

    //    std::cout << "back from recursion, will generate for assign now. \n"
    //    << std::flush; print(leftPath); print(valuePath);
    // Generate load and store here?
    //    rewriter.eraseOp(assign);
  } else {
    emitRemark(op->getLoc())
        << "Codegen for " << op->getName().getStringRef().str()
        << " unsupported!";
    //    std::cout << "\nI dont know how to do CodeGen for:"
    //              << op->getName().getStringRef().str() << std::flush;
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

      //      std::cout << "\nCodeGen for idxAcc!" << std::flush;
      path.push_back(idx.arg1());
      //      std::cout <<
      //      idx.arg0().getDefiningOp()->getName().getStringRef().str()
      //                <<
      //                idx.arg1().getDefiningOp()->getName().getStringRef().str()
      //                <<
      //                storeLocation.getDefiningOp()->getName().getStringRef().str()
      //                << std::flush;

      return codeGenStore(idx.arg0(), val, path, rewriter);
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
      return codeGen(idx.arg0(), path, rewriter);

    } else if (RiseBinaryOp binOp =
                   dyn_cast<RiseBinaryOp>(val.getDefiningOp())) {
      emitRemark(binOp.getLoc()) << "Codegen for binOp";

      //      std::cout << "\nCodeGen for binOp!" << std::flush;
      auto arg0 = codeGen(binOp.arg0(), {}, rewriter);
      auto arg1 = codeGen(binOp.arg1(), {}, rewriter);

      return rewriter.create<AddFOp>(val.getLoc(), arg0.getType(), arg0, arg1)
          .getResult();
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

    } else {
      //      std::cout << "\nI dont know how to do CodeGen for:"
      //                << val.getDefiningOp()->getName().getStringRef().str()
      //                << std::flush;
    }
  } else {
    // val is a BlockArg

    //    std::cout << "I have reached a BlockArg, should prob. do reverse now."
    //              << std::flush;
    //    print(path);
    //    path.push_back(val);
  }
  //  std::cout << "have nothing proper to return in CodeGen! I am returning the
  //  "
  //               "input.\n";
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

  return rewriter.create<LoadOp>(accessVal.getLoc(), accessVal, indexValues)
      .getResult();
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
  ValueRange valRange = ValueRange(indexValues);
  rewriter.create<StoreOp>(storeLoc.getLoc(), storeVal, storeLoc, indexValues);

  return;
}

/// This is obviously not really working.
/// For some Values it prints ints.
void mlir::rise::printPath(SmallVector<OutputPathType, 10> input) {
  struct {
    void operator()(Value val) {
      if (val.isa<OpResult>()) {
        std::cout << "val: "
                  << val.getDefiningOp()->getName().getStringRef().str();
      } else {
        std::cout << "blockArg, ";
      }
    }
    void operator()(Value *val) {
      if (val->isa<OpResult>()) {
        std::cout << "val: "
                  << val->getDefiningOp()->getName().getStringRef().str();
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
  //            << val.getDefiningOp()->getName().getStringRef().str() << " is:
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

LambdaOp mlir::rise::expandToLambda(mlir::Value value,
                                    PatternRewriter &rewriter) {

  Value appliedOp;
  if (isa<AddOp>(value.getDefiningOp())) {
    emitRemark(value.getLoc()) << "expanding add to lambda";

    LambdaOp newLambda;
    rewriter.setInsertionPoint(value.getDefiningOp());
    FunType funType = value.getType().dyn_cast<FunType>();

    newLambda = rewriter.create<LambdaOp>(value.getLoc(), funType);
    auto *entry = new Block();
    newLambda.region().push_back(entry);

    rewriter.setInsertionPointToStart(entry);

    // get the two summands
    SmallVector<Type, 4> argumentTypes = SmallVector<Type, 4>();
    argumentTypes.push_back(funType.getInput());
    argumentTypes.push_back(funType.getOutput().dyn_cast<FunType>().getInput());
    entry->addArguments(argumentTypes);

    appliedOp = rewriter.create<AddOp>(
        value.getLoc(), funType,
        DataTypeAttr::get(rewriter.getContext(),
                          cast<AddOp>(value.getDefiningOp()).t()));

    ApplyOp applyOp = rewriter.create<ApplyOp>(
        value.getLoc(), funType, appliedOp, entry->getArguments());
    rewriter.create<mlir::rise::ReturnOp>(value.getLoc(), ValueRange{applyOp});

    return newLambda;

  } else if (isa<MultOp>(value.getDefiningOp())) {
    emitRemark(value.getLoc()) << "expanding mul to lambda";

    LambdaOp newLambda;
    rewriter.setInsertionPoint(value.getDefiningOp());
    FunType funType = value.getType().dyn_cast<FunType>();

    newLambda = rewriter.create<LambdaOp>(value.getLoc(), funType);
    auto *entry = new Block();
    newLambda.region().push_back(entry);

    rewriter.setInsertionPointToStart(entry);

    // get the two factors
    SmallVector<Type, 4> argumentTypes = SmallVector<Type, 4>();
    argumentTypes.push_back(funType.getInput());
    argumentTypes.push_back(funType.getOutput().dyn_cast<FunType>().getInput());
    entry->addArguments(argumentTypes);

    appliedOp = rewriter.create<MultOp>(
        value.getLoc(), funType,
        DataTypeAttr::get(rewriter.getContext(),
                          cast<MultOp>(value.getDefiningOp()).t()));

    ApplyOp applyOp = rewriter.create<ApplyOp>(
        value.getLoc(), funType, appliedOp, entry->getArguments());
    rewriter.create<mlir::rise::ReturnOp>(value.getLoc(), ValueRange{applyOp});

    return newLambda;
  } else if (isa<ApplyOp>(value.getDefiningOp())) {
    //    std::cout << "expanding applied fun." << std::flush;
    //    LambdaOp newLambda;
    //    rewriter.setInsertionPoint(value.getDefiningOp());
    //    FunType funType = value.getType().dyn_cast<FunType>();
    //
    //    newLambda = rewriter.create<LambdaOp>(value.getLoc(), funType);
    //    auto *entry = new Block();
    //    newLambda.region().push_back(entry);
    //
    //    rewriter.setInsertionPointToStart(entry);
    //
    //
    //    return expandToLambda(cast<ApplyOp>(value.getDefiningOp()).fun(),
    //    rewriter);
    std::cout << " hi1: \n" << std::flush;

    LambdaOp newLambda;
    rewriter.setInsertionPoint(value.getDefiningOp());
    FunType funType = value.getType().dyn_cast<FunType>();

    newLambda = rewriter.create<LambdaOp>(value.getLoc(), funType);
    auto *entry = new Block();
    newLambda.region().push_back(entry);

    rewriter.setInsertionPointToStart(entry);
    std::cout << " hi1: \n" << std::flush;

    // get the two factors
    SmallVector<Type, 4> argumentTypes = SmallVector<Type, 4>();
    std::cout << " hi1: \n" << std::flush;

    argumentTypes.push_back(funType.getInput());
    std::cout << " hi1: \n" << std::flush;

    argumentTypes.push_back(funType.getOutput().dyn_cast<FunType>().getInput());
    std::cout << " hi1: \n" << std::flush;

    entry->addArguments(argumentTypes);

    //    appliedOp = rewriter.create<MultOp>(
    //        value.getLoc(), funType,
    //        DataTypeAttr::get(rewriter.getContext(),
    //                          cast<MultOp>(value.getDefiningOp()).t()));
    std::cout << " hi1: \n" << std::flush;
    appliedOp =
        cast<ApplyOp>(rewriter.clone(*(cast<ApplyOp>(value.getDefiningOp()))))
            .getResult();
    //    ApplyOp applyOp = rewriter.clone(*(cast<ApplyOp>(value)));
    std::cout << " hi2: \n" << std::flush;

    ApplyOp applyOp = rewriter.create<ApplyOp>(
        value.getLoc(), funType, appliedOp, entry->getArguments());
    rewriter.create<mlir::rise::ReturnOp>(value.getLoc(), ValueRange{applyOp});
    return newLambda;

  } else if (isa<MapOp>(value.getDefiningOp())) {
    std::cout << "expanding "
                 "map fun."
              << std::flush;

    //    LambdaOp newLambda;
    //    rewriter.setInsertionPoint(value.getDefiningOp());
    //    FunType funType = value.getType().dyn_cast<FunType>();
    //
    //    newLambda = rewriter.create<LambdaOp>(value.getLoc(), funType);
    //    auto *entry = new Block();
    //    newLambda.region().push_back(entry);
    //
    //    rewriter.setInsertionPointToStart(entry);
    //
    //    SmallVector<Type, 4> argumentTypes = SmallVector<Type, 4>();
    //    argumentTypes.push_back(funType.getInput());
    //    argumentTypes.push_back(funType.getOutput().dyn_cast<FunType>().getInput());
    //    entry->addArguments(argumentTypes);
    //
    //    rewriter.create<MapOp>(value.getLoc(), )
  } else {
    emitError(value.getLoc())
        << "Expanding " << value.getDefiningOp()->getName().getStringRef().str()
        << " to a Lambda is not supported.";
  }
}

/// gather all patterns
void mlir::populateRiseToImpConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ModuleToImp>(ctx);
}
//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// The pass:
void ConvertRiseToImperativePass::runOnModule() {
  auto module = getModule();

  // TODO: Initialize RiseTypeConverter here and use it below.
  //    std::unique_ptr<RiseTypeConverter> converter =
  //    makeRiseToStandardTypeConverter(&getContext());

  OwningRewritePatternList patterns;
  //    LinalgTypeConverter converter(&getContext());
  //    populateAffineToStdConversionPatterns(patterns, &getContext());
  //    populateLoopToStdConversionPatterns(patterns, &getContext());
  //    populateStdToLLVMConversionPatterns(converter, patterns);
  //    populateVectorToLLVMConversionPatterns(converter, patterns);
  //    populateLinalgToStandardConversionPatterns(patterns, &getContext());
  //    populateLinalgToLLVMConversionPatterns(converter, patterns,
  //    &getContext());
  populateRiseToImpConversionPatterns(patterns, &getContext());

  ConversionTarget target(getContext());

  //    target.addLegalDialect<StandardOpsDialect, AffineOpsDialect,
  //    mlir::loop::LoopOpsDialect>();
  target.addLegalOp<CallOp, FuncOp, ModuleOp, ModuleTerminatorOp, loop::ForOp,
                    ConstantIndexOp, AllocOp, LoadOp, StoreOp, AddFOp, MulFOp,
                    linalg::FillOp, mlir::ReturnOp, mlir::rise::LambdaOp,
                    mlir::rise::RiseIdxOp, mlir::rise::RiseBinaryOp,
                    mlir::rise::RiseFstIntermediateOp,
                    mlir::rise::RiseSndIntermediateOp,
                    mlir::rise::RiseZipIntermediateOp, mlir::rise::RiseAssignOp,
                    mlir::rise::ApplyOp, RiseContinuationTranslation>();
  //  target.addIllegalOp<RiseFunOp>();
  //  target.addDynamicallyLegalOp<RiseModuleOp>(
  //      [](RiseModuleOp op) { return op.lowered(); });
  //  target.markOpRecursivelyLegal<RiseModuleOp>();

  if (failed(applyPartialConversion(module, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::rise::createConvertRiseToImperativePass() {
  return std::make_unique<ConvertRiseToImperativePass>();
}

static PassRegistration<ConvertRiseToImperativePass>
    pass("convert-rise-to-imperative",
         "Compile all functional primitives of the rise dialect to imperative");

// Old Stuff:

// From Lambda:

// A Lambda has only one block
//  Block &block = lambda.region().front();
//  // For now start at the back and just find the first apply
//  ApplyOp lastApply;
//  for (auto op = block.rbegin(); op != block.rend(); op++) {
//    if (isa<ApplyOp>(*op)) {
//      lastApply = cast<ApplyOp>(*op);
//      break;
//    }
//  }
//
//  auto appliedFun = lastApply.getOperand(0);

// Finding the return from the chunk of rise IR
//  rise::ReturnOp returnOp = dyn_cast<rise::ReturnOp>(block.getTerminator());

//  rewriter.setInsertionPoint(&lambda.getParentRegion()->front(),
//  contLocation);

//  rewriter.setInsertionPointToStart(&lambda.region().front());

//    appliedFun.dropAllUses();
//    rewriter.eraseOp(appliedFun.getDefiningOp());
//
//    lastApply.getResult().dropAllUses();
//    rewriter.eraseOp(lastApply);
//
//    rewriter.eraseOp(returnOp);
//  // Now the Lambda is lowered. Note: We should prob. indicate this somehow
//  in
//  // the Op. Maybe in an Attribute Next inline the lambda where it is being
//  // applied.
//  lambda.region().front().getOperations().splice(
//      contLocation, lambda.region().front().getOperations());

// Splice moves operations (from,
//    lambda.getParentRegion()->front().getOperations().splice(
//        ++contLocation, lambda.region().front().getOperations(), addition);
//  std::cout << "\n empty: " << lambda.region().empty() << std::flush;
//
//    lambda.region().dropAllReferences();
//    lambda.region().front().dropAllReferences();
//    lambda.region().front().dropAllUses();
//    lambda.region().front().getOperations().clear();
//    lambda.region().front().eraseArgument(0);
//
//  std::cout << "\n empty: " << lambda.region().empty()
//            << " no uses?: " << lambda.region().front().use_empty()
//            << std::flush;
//  // We have to associate the block arguments of the first block with the
//  // arguments given to this function.
//  // Look in Block merge how two blocks are merged.
//  //  lambda.region().front().getArguments()
//  return lambda.getResult();
