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
  auto result = AccT(lastApply, outOp.getResult(), rewriter);
  // codegen here

  rewriter.setInsertionPointToEnd(&riseFun.getBody().back());
  auto newReturn =
      rewriter.create<mlir::ReturnOp>(returnOp.getLoc()); //,result);

  //  // We don't need the riseModule anymore
  //  riseFunOp.replaceAllUsesWith(callRiseFunOp);

  // replace output // TODO: put back in
//  riseFunOp.region().front().getArgument(0).replaceAllUsesWith(
//      riseFun.getBody().front().getArgument(0));

//  rewriter.eraseOp(riseFunOp);
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
    auto x1 = rewriter.create<LoadOp>(reductionFun.getLoc(), acc.getResult(),
                                      lowerBound.getResult());
    auto x2 = rewriter.create<LoadOp>(reductionFun.getLoc(), contArray,
                                      forLoop.getInductionVar());

    LambdaOp reductionLambda = dyn_cast<LambdaOp>(reductionFun.getDefiningOp());
    if (!reductionLambda) {
      reductionLambda = expandToLambda(reductionFun, rewriter);
    }

    ApplyOp lastApply;
    for (auto op = reductionLambda.region().front().rbegin();
         op != reductionLambda.region().front().rend(); op++) {
      if (isa<ApplyOp>(*op)) {
        lastApply = cast<ApplyOp>(*op);
        break;
      }
    }
    Substitute(reductionLambda, {x1.getResult(), x2.getResult()});

    rewriter.setInsertionPointAfter(x2);
    auto lambdaResult = AccT(lastApply, {}, rewriter);
    rewriter.setInsertionPointAfter(lambdaResult.getDefiningOp());

    auto storing =
        rewriter.create<StoreOp>(reductionFun.getLoc(), lambdaResult,
                                 acc.getResult(), lowerBound.getResult());

    return acc.getResult();

  } else if (isa<MapOp>(appliedFun)) {
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
    ArrayRef<int64_t> shape =
        contArray.getType().dyn_cast<MemRefType>().getShape().drop_back(1);

    MemRefType indexOpResult =
        MemRefType::get(shape, FloatType::getF32(rewriter.getContext()));
    auto xi =
        rewriter.create<RiseIdxOp>(loc, indexOpResult, contArray, forLoop.getInductionVar());

    // create Apply for Lambda now, and create other idx for the out.
    // start AccT for Apply of Lambda and add case for Lamda



    //    // This LoadOp should not be created here.
    //    auto x1 = rewriter.create<LoadOp>(appliedFun->getLoc(), contArray,
    //                                      forLoop.getInductionVar());

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

  } else if (isa<AddOp>(appliedFun)) {
    auto summand0 = applyOperands.pop_back_val();
    auto summand1 = applyOperands.pop_back_val();

    auto contSummand0 = ConT(summand0, rewriter.getInsertionPoint(), rewriter);
    auto contSummand1 = ConT(summand1, rewriter.getInsertionPoint(), rewriter);

    auto newAddOp = rewriter.create<AddFOp>(appliedFun->getLoc(), contSummand0,
                                            contSummand1);
    return newAddOp;

  } else if (isa<MultOp>(appliedFun)) {
    auto factor0 = applyOperands.pop_back_val();
    auto factor1 = applyOperands.pop_back_val();

    auto contFactor0 = ConT(factor0, rewriter.getInsertionPoint(), rewriter);
    auto contFactor1 = ConT(factor1, rewriter.getInsertionPoint(), rewriter);

    auto newMultOp =
        rewriter.create<MulFOp>(appliedFun->getLoc(), contFactor0, contFactor1);
    return newMultOp;
  } else if (isa<ApplyOp>(appliedFun)) {
    emitError(appliedFun->getLoc()) << "We should never get here";
    //    auto tmp = AccT(expression, cast<ApplyOp>(appliedFun), rewriter);

  } else {
    emitRemark(appliedFun->getLoc())
        << "lowering op: " << appliedFun->getName() << " not yet supported.";
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
  LambdaOp newLambda;
  rewriter.setInsertionPoint(value.getDefiningOp());
  FunType funType = value.getType().dyn_cast<FunType>();

  newLambda = rewriter.create<LambdaOp>(value.getLoc(), funType);
  auto *entry = new Block();
  newLambda.region().push_back(entry);

  rewriter.setInsertionPointToStart(entry);

  Value appliedOp;
  if (isa<AddOp>(value.getDefiningOp())) {
    // get the two summands
    SmallVector<Type, 4> argumentTypes = SmallVector<Type, 4>();
    argumentTypes.push_back(funType.getInput());
    argumentTypes.push_back(funType.getOutput().dyn_cast<FunType>().getInput());
    entry->addArguments(argumentTypes);

    appliedOp = rewriter.create<AddOp>(
        value.getLoc(), funType,
        DataTypeAttr::get(rewriter.getContext(),
                          cast<AddOp>(value.getDefiningOp()).t()));
  } else if (isa<MultOp>(value.getDefiningOp())) {
    // get the two factors
    SmallVector<Type, 4> argumentTypes = SmallVector<Type, 4>();
    argumentTypes.push_back(funType.getInput());
    argumentTypes.push_back(funType.getOutput().dyn_cast<FunType>().getInput());
    entry->addArguments(argumentTypes);

    appliedOp = rewriter.create<MultOp>(
        value.getLoc(), funType,
        DataTypeAttr::get(rewriter.getContext(),
                          cast<MultOp>(value.getDefiningOp()).t()));
  } else {
    emitError(value.getLoc())
        << "Expanding this operation to a Lambda is not supported.";
  }

  ApplyOp applyOp = rewriter.create<ApplyOp>(value.getLoc(), funType, appliedOp,
                                             entry->getArguments());
  rewriter.create<mlir::rise::ReturnOp>(value.getLoc(), ValueRange{applyOp});

  return newLambda;
}

/// Continuation Translation
mlir::Value mlir::rise::ConT(mlir::Value contValue,
                             Block::iterator contLocation,
                             PatternRewriter &rewriter) {
  Location loc = contValue.getLoc();
  auto oldInsertPoint = rewriter.saveInsertionPoint();

  if (isa<LiteralOp>(contValue.getDefiningOp())) {
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
    }
  } else if (isa<LambdaOp>(contValue.getDefiningOp())) {
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
                    linalg::FillOp, mlir::ReturnOp, mlir::rise::RiseIdxOp,
                    RiseContinuationTranslation>();
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
