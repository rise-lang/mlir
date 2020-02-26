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

namespace mlir {
namespace rise {
void AccT(Block *expression, mlir::Value output, PatternRewriter &rewriter);
mlir::Value ConT(mlir::Value contValue,
                        PatternRewriter &rewriter);
} // namespace rise
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module,
                                           LLVM::LLVMDialect *llvmDialect);
} // namespace mlir

#endif // MLIR_CONVERTRISETOIMPERATIVE_H
