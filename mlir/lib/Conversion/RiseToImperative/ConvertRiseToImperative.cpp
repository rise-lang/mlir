//
// Created by martin on 26/11/2019.
//

#include "mlir/Conversion/RiseToImperative/ConvertRiseToImperative.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
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
#include <mlir/EDSC/Builders.h>

using namespace mlir;
using namespace mlir::rise;


namespace {
struct ConvertRiseToImperativePass : public ModulePass<ConvertRiseToImperativePass> {
    void runOnModule() override;
    };
} // namespace

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

///Apply
struct ApplyToImpLowering : public OpRewritePattern<ApplyOp> {
    using OpRewritePattern<ApplyOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(ApplyOp applyOp,
                                       PatternRewriter &rewriter) const override;
};

PatternMatchResult ApplyToImpLowering::matchAndRewrite(ApplyOp applyOp,
                                   PatternRewriter &rewriter) const {
    MLIRContext *context = rewriter.getContext();
    Location loc = applyOp.getLoc();
    //TODO: do
    emitError(loc) << "yoyoyo";
    return matchSuccess();

}





struct ModuleToImp : public OpRewritePattern<RiseModuleOp> {
    using OpRewritePattern<RiseModuleOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(RiseModuleOp moduleOp,
                                       PatternRewriter &rewriter) const override;
};

PatternMatchResult ModuleToImp::matchAndRewrite(RiseModuleOp moduleOp, PatternRewriter &rewriter) const {
    rewriter.startRootUpdate(moduleOp);

    MLIRContext *context = rewriter.getContext();
    Location loc = moduleOp.getLoc();
    Region &riseRegion = moduleOp.region();
//    Region::BlockListType &blocklist = riseRegion.getBlocks();
    Block &block = riseRegion.getBlocks().front();     // A rise region only has one block
//    Region::BlockListType::reverse_iterator blockIterator = blocklist.rbegin();

    /// For now start at the back and just find the first apply
    ApplyOp lastApply;
    for (auto op = block.rbegin(); op != block.rend(); op++) {
        if (isa<ApplyOp>(*op)) {
            lastApply = cast<ApplyOp>(*op);
            emitRemark(lastApply.getLoc()) << "apply found";
            break;
        }
    }
    auto operations = &riseRegion.getBlocks().front().getOperations();
//    operations->back().erase(); //removes the return - do I even want this?
    AccT(&block, lastApply.getResult(), rewriter);


    /// We want to take the last operation (before the return?) and start translation from there.

    //TODO: We get an error because this requires us to replace the root op of the rewrite.

//    auto newModule = rewriter.create<ModuleOp>(moduleOp.getLoc());
//    BlockAndValueMapping *mapping;
//    moduleOp.region().cloneInto(&newModule.getBodyRegion());
//    rewriter.replaceOp(moduleOp, newModul);

//    auto loc2 = mlir::edsc::ScopedContext::getLocation(); //Not sure how to use this

    rewriter.setInsertionPointToEnd(&moduleOp.region().front());
    rewriter.create<mlir::rise::ReturnOp>(moduleOp.getLoc());
    /// What is even happening here, do I have to insert this op now into the block op list?
    /// How can I get a handle to it then?


    /// Im sure this is not the correct thing to do. However this way I can debug my stuff now.
    moduleOp.ensureTerminator(moduleOp.region(), rewriter, rewriter.getUnknownLoc()); //The location here is wrong.
    rewriter.finalizeRootUpdate(moduleOp);





    /// We can get all operations inside this rise module and work on them.
    emitRemark(loc) << "I found the following operations:";
    for (auto &block : riseRegion.getBlocks()) {
        for (auto &operation : block.getOperations()) {
            emitRemark(operation.getLoc()) << "op: " << operation.getName().getStringRef();
        }
    }

    return matchSuccess();
}

/// Acceptor Translation
/// expression (List of Operations) for Acceptor Translation    - E in paper
/// output "pointer" for Acceptor Translation                   - A in paper
/// returns a (partially) lowered expression (list of operations)
/// Using the existing OpListType should make things fairly straight forward.
void mlir::rise::AccT(Block *expression, mlir::Value output, PatternRewriter &rewriter) {
    Block::OpListType &operations = expression->getOperations();
    emitRemark((++operations.rbegin())->getLoc()) << "starting Acceptor Translation. Output: " << output.getLoc() << " last op: " << (++operations.rbegin())->getName();



    ApplyOp apply = cast<ApplyOp>(*(++operations.rbegin()));
//    operations.removeNodeFromList(apply);

    /// find applied function
    Operation* appliedFun; //= &expression->back(); //must not be undefined. Just initialized with the last one. Prob wrong approach
    for (auto &op : operations) {
        if (op.getResult(0) == apply.fun()) {
            emitRemark(op.getLoc()) << "found the applied function";
            appliedFun = &op;

            break;
        }
    }
//    operations.removeNodeFromList(appliedFun);
    emitRemark(appliedFun->getLoc()) << "The applied fun is " << appliedFun->getName();

    if (isa<ReduceOp>(appliedFun)) {
        auto n = appliedFun->getAttrOfType<NatAttr>("n");
        auto s = appliedFun->getAttrOfType<DataTypeAttr>("s");
        auto t = appliedFun->getAttrOfType<DataTypeAttr>("t");
        auto reductionFun = apply.getOperand(1);
        auto initializer = apply.getOperand(2);
        auto array = apply.getOperand(3);
        emitRemark(appliedFun->getLoc()) << "Attributes: n:" << n.getValue().getIntValue() << " s: " << s.getValue().getKind() << " t: " << t.getValue().getKind() << " initializer: " << initializer.getType();

        /// Here I know what to do with the reduce and where to find the information for its codegen. -> look up in apply

        /// The lower upper bounds and the step also have to be Values.
        Location loc = apply.getLoc();
        rewriter.eraseOp(apply);

        rewriter.setInsertionPointToEnd(expression);
        auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto upperBound = rewriter.create<ConstantIndexOp>(loc, 3);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
//        auto forLoop = rewriter.create<mlir::loop::ForOp>(loc , lowerBound, upperBound, step);
        /// This is not working like this right now. The operation is added to the IR I think, but I can't add it to my list of operations.
//        forLoop.region().getBlocks().clear();
//        rewriter.inlineRegionBefore(expression->getParent(), forLoop.region(), forLoop.region().end());
//        rewriter.inlineRegionBefore(forLoop.region(), expression);
//        rewriter.inlineRegionBefore(forLoop.region(), *expression->getParent(), expression->getParent()->end());
//        expression->push_back(forLoop);
//        expression->push_back(forLoop);
    } else if (isa<MapOp>(appliedFun)){

    } else {
        emitRemark(appliedFun->getLoc()) << "lowering op: " << appliedFun->getName() << " not yet supported.";
    }

    //FIXME: This seems to work the way I want it, at least when looking at my emitted remarks. However, my IR is not changed.

    //goal:
    //func @main() {
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
Block::OpListType* mlir::rise::ConT(Block *expression, mlir::Value output, PatternRewriter &rewriter) {
//    emitRemark(loc) << "starting Continuation Translation";
    return &expression->getOperations();
}



///gather all patterns
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

    //TODO: Initialize RiseTypeConverter here and use it below.
//    std::unique_ptr<RiseTypeConverter> converter = makeRiseToStandardTypeConverter(&getContext());

    OwningRewritePatternList patterns;
//    LinalgTypeConverter converter(&getContext());
//    populateAffineToStdConversionPatterns(patterns, &getContext());
//    populateLoopToStdConversionPatterns(patterns, &getContext());
//    populateStdToLLVMConversionPatterns(converter, patterns);
//    populateVectorToLLVMConversionPatterns(converter, patterns);
//    populateLinalgToStandardConversionPatterns(patterns, &getContext());
//    populateLinalgToLLVMConversionPatterns(converter, patterns, &getContext());
    populateRiseToImpConversionPatterns(patterns, &getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<FuncOp, ModuleOp, ModuleTerminatorOp>();
//    target.addDynamicallyLegalOp<FuncOp>(
//            [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
//    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

    //TODO: Add our TypeConverter as last argument
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
