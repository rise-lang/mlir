//
// Created by martin on 26/11/2019.
//

#ifndef MLIR_CONVERTRISETOIMPERATIVE_H
#define MLIR_CONVERTRISETOIMPERATIVE_H

#include "mlir/Dialect/Rise/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"

#include "mlir/Dialect/Rise/variant.hpp"

namespace mlir {
namespace rise {
// std::variant would be nice to have here. Look at how this works. I need a sum
// type for my path, I think
using OutputPathType = mpark::variant<int, bool, Value>;
mlir::Value AccT(ApplyOp apply, Value out,
                 PatternRewriter &rewriter);
mlir::Value ConT(mlir::Value contValue, Block::iterator contLocation,
                 PatternRewriter &rewriter);
SmallVector<OutputPathType, 10> codeGen(Operation *op, SmallVector<OutputPathType,10> path);
SmallVector<OutputPathType, 10> codeGen(Value val, SmallVector<OutputPathType,10> path);

void Substitute(LambdaOp lambda, llvm::SmallVector<Value, 10> args);
LambdaOp expandToLambda(mlir::Value value, PatternRewriter &rewriter);
void print(SmallVector<OutputPathType , 10> input);

} // namespace rise
} // namespace mlir

#endif // MLIR_CONVERTRISETOIMPERATIVE_H
