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

struct ModuleToImp : public OpRewritePattern<RiseModuleOp> {
  using OpRewritePattern<RiseModuleOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(RiseModuleOp moduleOp,
                                     PatternRewriter &rewriter) const override;
};

PatternMatchResult
ModuleToImp::matchAndRewrite(RiseModuleOp moduleOp,
                             PatternRewriter &rewriter) const {
  std::cout << "\n the Imperative pass is up and running \n";

  MLIRContext *context = rewriter.getContext();
  Location loc = moduleOp.getLoc();
  Region &riseRegion = moduleOp.region();

  // create mlir function for the given rise module and inline all rise
  // operations
  auto riseFun = rewriter.create<FuncOp>(loc, StringRef("riseFun"),
                                         FunctionType::get({}, {}, context),
                                         ArrayRef<NamedAttribute>{});
  rewriter.inlineRegionBefore(moduleOp.region(), riseFun.getBody(),
                              riseFun.getBody().begin());

  // We don't need the riseModule anymore
  rewriter.eraseOp(moduleOp);

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

  // Finding the return
  rise::ReturnOp returnOp;
  for (auto op = block.rbegin(); op != block.rend(); op++) {
    if (isa<rise::ReturnOp>(*op)) {
      returnOp = cast<rise::ReturnOp>(*op);
      break;
    }
  }

  // Start translation to imperative
  AccT(&block, lastApply.getResult(), rewriter);

  //  // printing all operations inside riseFun
  //  emitRemark(loc) << "I found the following operations:";
  //  for (auto &block : riseFun.getBody().getBlocks()) {
  //    for (auto &operation : block.getOperations()) {
  //                  emitRemark(operation.getLoc()) << "op: " <<
  //                  operation.getName().getStringRef();
  //    }
  //  }

  return matchSuccess();
}

/// Acceptor Translation
/// expression (List of Operations) for Acceptor Translation    - E in paper
/// output "pointer" for Acceptor Translation                   - A in paper
/// returns a (partially) lowered expression (list of operations)
/// Using the existing OpListType should make things fairly straight forward.
void mlir::rise::AccT(Block *expression, mlir::Value output,
                      PatternRewriter &rewriter) {
  Block::OpListType &operations = expression->getOperations();

  // This assumes that apply is the second to last operation
  ApplyOp apply = cast<ApplyOp>(*(++operations.rbegin()));
  rewriter.setInsertionPoint(apply);

  /// find applied function
  Operation *appliedFun;
  for (auto &op : operations) {
    if (op.getResult(0) == apply.fun()) {
      appliedFun = &op;
      break;
    }
  }

  emitRemark(appliedFun->getLoc())
      << "The applied fun is " << appliedFun->getName();

  // Check which translation we do. TODO: move this to its own function
  if (isa<ReduceOp>(appliedFun)) {
    auto n = appliedFun->getAttrOfType<NatAttr>("n");
    auto s = appliedFun->getAttrOfType<DataTypeAttr>("s");
    auto t = appliedFun->getAttrOfType<DataTypeAttr>("t");
    auto reductionFun = apply.getOperand(1);
    auto initializer = apply.getOperand(2);
    auto array = apply.getOperand(3);

    Location loc = apply.getLoc();

    /// TODO: Think: At the point where we add the continuation we could also
    // just make a recursive call to the translation.
    // Leave it like this for now

    // Add Continuation for array. Initialization of it will also come from
    // this in case of an array literal
    auto contArray = rewriter.create<RiseContinuationTranslation>(
        array.getLoc(),
        MemRefType::get(ArrayRef<int64_t>{32},
                        FloatType::getF32(rewriter.getContext())),
        array);

    // Add Continuation for init
    auto contInit = ConT(initializer, rewriter);

    // Accumulator for Reduction
    auto acc = rewriter.create<AllocOp>(
        appliedFun->getLoc(),
        MemRefType::get(ArrayRef<int64_t>{0},
                        FloatType::getF32(rewriter.getContext())));

    // Add alloc for accumulator
    // add linalg.fill for input array
    auto lowerBound = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 0);
    auto upperBound = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 3);
    auto step = rewriter.create<ConstantIndexOp>(appliedFun->getLoc(), 1);

    auto forLoop =
        rewriter.create<mlir::loop::ForOp>(loc, lowerBound, upperBound, step);

    // create operations for addition inside the loop
    rewriter.setInsertionPointToStart(forLoop.getBody());
    auto x1 = rewriter.create<LoadOp>(reductionFun.getLoc(), acc.getResult(),
                                      lowerBound.getResult());
    auto x2 =
        rewriter.create<LoadOp>(reductionFun.getLoc(), contArray.getResult(),
                                forLoop.getInductionVar());
    auto addition = rewriter.create<AddFOp>(reductionFun.getLoc(),
                                            x1.getResult(), x2.getResult());
    auto storing =
        rewriter.create<StoreOp>(reductionFun.getLoc(), addition.getResult(),
                                 acc.getResult(), lowerBound.getResult());

    // remomve apply
    apply.replaceAllUsesWith(acc.getResult());
    rewriter.eraseOp(apply);

    // remove reduce
    appliedFun->dropAllUses();
    rewriter.eraseOp(appliedFun);

    // remove reduction Function TODO: reductionFun is not yet realized with
    //  cont
    reductionFun.dropAllUses();
    rewriter.eraseOp(reductionFun.getDefiningOp());
  } else if (isa<MapOp>(appliedFun)) {

  } else {
    emitRemark(appliedFun->getLoc())
        << "lowering op: " << appliedFun->getName() << " not yet supported.";
  }

  // FIXME: This seems to work the way I want it, at least when looking at my
  // emitted remarks. However, my IR is not changed.

  // goal:
  // func @main() {
  //  %array = alloc() : memref<4xf32>
  //  %init = alloc() : memref<1xf32>
  //  %cst_0 = constant 0 : index
  //
  //  %lb = constant 0 : index
  //  %ub = constant 4 : index //half open index, so 4 iterations
  //  %step = constant 1 : index
  //  loop.for %i = %lb to %ub step %step {
  //    %elem = load %array[%i] : memref<4xf32>
  //    %acc = load %init[%cst_0] : memref<1xf32>
  //    %res = addf %acc, %elem : f32
  //    store %res, %init[%cst_0] : memref<1xf32>
  //  }
  //  return
  //
}

/// Continuation Translation
mlir::Value mlir::rise::ConT(mlir::Value contValue, PatternRewriter &rewriter) {
  std::cout << "\n 0\n";
  //
  //  if (LiteralOp op = dyn_cast<LiteralOp>(contValue.getDefiningOp())) {
  //    std::cout << "\n 1\n" << op.literalAttr().getType().getKind() << " and "
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
  rewriter.create<RiseContinuationTranslation>(
      contValue.getLoc(), FloatType::getF32(rewriter.getContext()), contValue);
  return contValue;
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
  target.addLegalOp<FuncOp, ModuleOp, ModuleTerminatorOp, loop::ForOp,
                    ConstantIndexOp, AllocOp, LoadOp, StoreOp, AddFOp,
                    RiseContinuationTranslation>();
  target.addIllegalOp<RiseModuleOp>();
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
