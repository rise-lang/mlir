//
// Created by martin on 26/11/2019.
//

#include "mlir/Conversion/RiseToStandard/ConvertRiseToStandard.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"
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
struct ConvertRiseToStandardPass : public ModulePass<ConvertRiseToStandardPass> {
    void runOnModule() override;
};
} // namespace


Type RiseTypeConverter::convertType(Type t) {
    if (t.getKind() == RiseTypeKind::RISE_INT) {
        return convertIntType(t.cast<Int>());
    }
    return t;
}

Type RiseTypeConverter::convertIntType(Int type) {
//    StandardOpsDialect::
}


//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

///Literal
struct LiteralLowering : public OpRewritePattern<LiteralOp> {
    using OpRewritePattern<LiteralOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(LiteralOp literalOp,
                                       PatternRewriter &rewriter) const override;
};

PatternMatchResult
LiteralLowering::matchAndRewrite(LiteralOp literalOp, PatternRewriter &rewriter) const {
    Location loc = literalOp.getLoc();
    LiteralAttr literalAttr = literalOp.getAttrOfType<LiteralAttr>("literal");
    if (!literalAttr) {
        printf("we got a nullptr here, captain! \n");
    }
    ///conversion to itself
//    rewriter.create<LiteralOp>(loc, literalAttr.getType(), literalAttr);

    switch (literalAttr.getType().getKind()) {
        default: {
            emitError(loc) << "could not lower rise.literal";
            return matchFailure();
        }
        case RiseTypeKind::RISE_INT: {
            int32_t value = std::stoi(literalAttr.getValue());
            rewriter.create<ConstantIntOp>(loc, value, IntegerType::get(32, rewriter.getContext())); //why is this getting such a strange SSA id?
            break;
        }
        case RiseTypeKind::RISE_FLOAT: {
            APFloat value = APFloat(std::stof(literalAttr.getValue()));
            rewriter.create<ConstantFloatOp>(loc, value, FloatType::getF32(rewriter.getContext()));
            break;
        }
    }

    rewriter.eraseOp(literalOp);
    return matchSuccess();
}

///Lambda
struct LambdaLowering : public OpRewritePattern<LambdaOp> {
    using OpRewritePattern<LambdaOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(LambdaOp lambdaOp,
                                       PatternRewriter &rewriter) const override;
};

PatternMatchResult
LambdaLowering::matchAndRewrite(LambdaOp lambdaOp, PatternRewriter &rewriter) const {
    MLIRContext *context = rewriter.getContext();
    Location loc = lambdaOp.getLoc();

    FunctionType funType = FunctionType::get(lambdaOp.getType().cast<FunType>().getInput(), lambdaOp.getType().cast<FunType>().getInput(), context);
    FuncOp fun = rewriter.create<FuncOp>(loc, "testFun", funType, ArrayRef<NamedAttribute>{});
    Block *funBody = fun.addEntryBlock();

    //Adding the region of the lambdaOp to the FuncOp. Enclosed ops are handled separately
    rewriter.inlineRegionBefore(lambdaOp.region(), *funBody->getParent(), funBody->getParent()->end());

    rewriter.eraseOp(lambdaOp);

    return matchSuccess();
}

///Return
struct ReturnLowering : public OpRewritePattern<mlir::rise::ReturnOp> {
    using OpRewritePattern<mlir::rise::ReturnOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(mlir::rise::ReturnOp returnOp,
                                       PatternRewriter &rewriter) const override;
};

PatternMatchResult
ReturnLowering::matchAndRewrite(mlir::rise::ReturnOp returnOp, PatternRewriter &rewriter) const {
    MLIRContext *context = rewriter.getContext();

    rewriter.replaceOpWithNewOp<mlir::ReturnOp>(returnOp);
    
//
//    Location loc = returnOp.getLoc();
//
//    mlir::ReturnOp stdReturn = rewriter.create<mlir::ReturnOp>(loc);
////    rewriter.
//    rewriter.eraseOp(returnOp);

    return matchSuccess();
}





///Apply
struct ApplyLowering : public OpRewritePattern<ApplyOp> {
    using OpRewritePattern<ApplyOp>::OpRewritePattern;

    PatternMatchResult matchAndRewrite(ApplyOp applyOp,
                                       PatternRewriter &rewriter) const override;
};

PatternMatchResult ApplyLowering::matchAndRewrite(ApplyOp applyOp,
                                   PatternRewriter &rewriter) const {
    MLIRContext *context = rewriter.getContext();
    Location loc = applyOp.getLoc();
    //TODO: do
    return matchSuccess();

}

///gather all patterns
void mlir::populateRiseToStdConversionPatterns(
        OwningRewritePatternList &patterns, MLIRContext *ctx) {
    patterns.insert<LiteralLowering, LambdaLowering, ReturnLowering>(ctx);
}


//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

///// Create an instance of LLVMTypeConverter in the given context.
//static std::unique_ptr<RiseTypeConverter>
//makeRiseToStandardTypeConverter(MLIRContext *context) {
//    return std::make_unique<RiseTypeConverter>(context);
//}


/// The pass:
void ConvertRiseToStandardPass::runOnModule() {
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
    populateRiseToStdConversionPatterns(patterns, &getContext());

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
mlir::rise::createConvertRiseToStandardPass() {
    return std::make_unique<ConvertRiseToStandardPass>();
}

static PassRegistration<ConvertRiseToStandardPass>
        pass("convert-rise-to-standard",
             "Convert the operations from the rise dialect into the standard dialect");
