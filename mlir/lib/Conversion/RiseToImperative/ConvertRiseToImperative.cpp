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
  auto riseFun = rewriter.create<FuncOp>(
      loc, StringRef("riseFun"),
      FunctionType::get({}, riseFunOp.getResult().getType(), context),
      ArrayRef<NamedAttribute>{});
  rewriter.inlineRegionBefore(riseFunOp.region(), riseFun.getBody(),
                              riseFun.getBody().begin());

  rewriter.setInsertionPointToStart(&riseFunOp.getParentRegion()->front());
  auto callRiseFunOp = rewriter.create<CallOp>(riseFunOp.getLoc(), riseFun);

  // We don't need the riseModule anymore
  riseFunOp.replaceAllUsesWith(callRiseFunOp);
  rewriter.eraseOp(riseFunOp);

  // The function has only one block, as a rise module can only have one block.
  Block &block = riseFun.getBody().front();
  // For now start at the back and just find the first apply
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
  auto result = AccT(lastApply, rewriter);

  // Replace rise.return (leaving the rise module) with a return for the funcOp
  rewriter.setInsertionPoint(returnOp);
  auto newReturn = rewriter.create<mlir::ReturnOp>(returnOp.getLoc(), result);

  rewriter.eraseOp(returnOp);

  return matchSuccess();
}

/// Acceptor Translation
/// expression (List of Operations) for Acceptor Translation    - E in paper
/// output "pointer" for Acceptor Translation                   - A in paper
/// returns a lowered expression (list of operations)
/// Using the existing OpListType should make things fairly straight forward.
mlir::Value mlir::rise::AccT(ApplyOp apply, PatternRewriter &rewriter) {
  Operation *appliedFun = apply.getOperand(0).getDefiningOp();

  // If functions are applied partially i.e. appliedFun is an ApplyOp we recurse
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
    auto contArray = ConT(array, rewriter);

    // Add Continuation for init
    auto contInit = ConT(initializer, rewriter);

    rewriter.setInsertionPoint(apply);
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

    // TODO: This addition is hardcoded. Add Transl
    auto addition = rewriter.create<AddFOp>(reductionFun.getLoc(),
                                            x1.getResult(), x2.getResult());
    auto storing =
        rewriter.create<StoreOp>(reductionFun.getLoc(), addition.getResult(),
                                 acc.getResult(), lowerBound.getResult());

    // remomve applies
    while (applyStack.size()) {
      applyStack.back().getOperation()->dropAllUses();
      rewriter.eraseOp(applyStack.pop_back_val());
    }

    // remove reduce
    appliedFun->dropAllUses();
    rewriter.eraseOp(appliedFun);

    // remove reduction Function TODO: reductionFun is not yet realized with
    reductionFun.dropAllUses();
    rewriter.eraseOp(reductionFun.getDefiningOp());

    return acc.getResult();

  } else if (isa<MapOp>(appliedFun)) {
    //     For now we treat all maps as mapSeqs
    auto n = appliedFun->getAttrOfType<NatAttr>("n");
    auto s = appliedFun->getAttrOfType<DataTypeAttr>("s");
    auto t = appliedFun->getAttrOfType<DataTypeAttr>("t");

    // TODO: LambdaOp not translated yet
    auto f = applyOperands.pop_back_val();
    auto array = applyOperands.pop_back_val();

    auto contArray = ConT(array, rewriter);

    rewriter.setInsertionPoint(apply);
    auto lowerBound = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 0);
    auto upperBound = rewriter.create<ConstantIndexOp>(
        appliedFun->getLoc(), n.getValue().getIntValue());
    auto step = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 1);

    auto forLoop =
        rewriter.create<mlir::loop::ForOp>(loc, lowerBound, upperBound, step);

    // create operations for addition inside the loop
    rewriter.setInsertionPointToStart(forLoop.getBody());
    auto x1 = rewriter.create<LoadOp>(appliedFun->getLoc(), contArray,
                                      forLoop.getInductionVar());

    // TODO: This addition is hardcoded. Add Transl
    auto addition = rewriter.create<AddFOp>(appliedFun->getLoc(),
                                            x1.getResult(), x1.getResult());

    auto storing =
        rewriter.create<StoreOp>(appliedFun->getLoc(), addition.getResult(),
                                 contArray, forLoop.getInductionVar());

    // Finished, erase all applies:
    while (applyStack.size()) {
      applyStack.back().getOperation()->dropAllUses();
      rewriter.eraseOp(applyStack.pop_back_val());
    }
    // remove LambdaOp
    f.getDefiningOp()->dropAllUses();
    rewriter.eraseOp(f.getDefiningOp());

    // remove MapOp
    appliedFun->dropAllUses();
    rewriter.eraseOp(appliedFun);

    return contArray;

  } else if (isa<ApplyOp>(appliedFun)) {
    emitError(appliedFun->getLoc()) << "We should never get here";
    //    auto tmp = AccT(expression, cast<ApplyOp>(appliedFun), rewriter);

  } else {
    emitRemark(appliedFun->getLoc())
        << "lowering op: " << appliedFun->getName() << " not yet supported.";
  }
}

/// Continuation Translation
mlir::Value mlir::rise::ConT(mlir::Value contValue, PatternRewriter &rewriter) {
  Location loc = contValue.getLoc();

  StringRef literalValue =
      dyn_cast<LiteralOp>(contValue.getDefiningOp()).literalAttr().getValue();
  // This should of course check for an int or float type
  // However the casting for some reason does not work. I'll hardcode it for now

  //  TODO: I should be doing this. But this does not work
  //  std::cout << "\nTest printing";
  //  if (LiteralOp op = dyn_cast<LiteralOp>(contValue.getDefiningOp())) {
  //    if (op.literalAttr().getType().kindof(RiseTypeKind::RISE_FLOAT)) {
  //      std::cout << "\nHouston, we have a Float Literal";
  //    }
  //  }

  if (literalValue == "0") {
    rewriter.setInsertionPointAfter(contValue.getDefiningOp());

    // Done. Erase Op and return
    contValue.getDefiningOp()->dropAllUses();
    rewriter.eraseOp(contValue.getDefiningOp());

    return rewriter.create<ConstantFloatOp>(
        loc, llvm::APFloat(0.0f), FloatType::getF32(rewriter.getContext()));

    //      rewriter.create<RiseContinuationTranslation>(
    //          contValue.getLoc(), FloatType::getF32(rewriter.getContext()),
    //          contValue);
    // This should check for an array type
  } else if (literalValue == "[5,5,5,5]") {
    rewriter.setInsertionPointAfter(contValue.getDefiningOp());

    auto array = rewriter.create<AllocOp>(
        loc, MemRefType::get(ArrayRef<int64_t>{4},
                             FloatType::getF32(rewriter.getContext())));
    // For now just fill the array with one value
    auto filler = rewriter.create<ConstantFloatOp>(
        loc, llvm::APFloat(5.0f), FloatType::getF32(rewriter.getContext()));

    rewriter.create<linalg::FillOp>(loc, array.getResult(), filler.getResult());

    // Done. erase Op and return
    contValue.getDefiningOp()->dropAllUses();
    rewriter.eraseOp(contValue.getDefiningOp());
    return array.getResult();

    //    return rewriter.create<RiseContinuationTranslation>(
    //        contValue.getLoc(),
    //        MemRefType::get(ArrayRef<int64_t>{32},
    //                        FloatType::getF32(rewriter.getContext())),
    //        contValue);
  } else {
    emitError(contValue.getLoc())
        << "can not perform continuation "
           "translation for "
        << contValue.getDefiningOp()->getName().getStringRef().str();
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
                    ConstantIndexOp, AllocOp, LoadOp, StoreOp, AddFOp,
                    linalg::FillOp, mlir::ReturnOp,
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
