//
// Created by martin on 26/11/2019.
//

#include "mlir/Conversion/RiseToStandard/ConvertRiseToStandard.h"
#include "mlir/Dialect/Rise/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include <iostream>

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





///module lowering (to be rise.module)
struct ModuleToImp : public OpRewritePattern<RiseModuleOp> {
    using OpRewritePattern<RiseModuleOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(RiseModuleOp moduleOp,
                                       PatternRewriter &rewriter) const override;
};

PatternMatchResult ModuleToImp::matchAndRewrite(RiseModuleOp moduleOp, PatternRewriter &rewriter) const {
    MLIRContext *context = rewriter.getContext();
    Location loc = moduleOp.getLoc();
    Region &riseRegion = moduleOp.region();
//    Region::BlockListType &blocklist = riseRegion.getBlocks();
//    Region::BlockListType::reverse_iterator blockIterator = blocklist.rbegin();

    /// We can get all operations inside this rise module and work on them.
    emitRemark(loc) << "I found the following operations:";
    for (auto &block : riseRegion.getBlocks()) {
        for (auto &operation : block.getOperations()) {
            emitRemark(operation.getLoc()) << "op: " << operation.getName().getStringRef();
        }
    }
    /// We want to take the last operation (before the return?) and start translation from there.

    //TODO: We get an error because this requires us to replace the root op of the rewrite.
    return matchSuccess();
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
