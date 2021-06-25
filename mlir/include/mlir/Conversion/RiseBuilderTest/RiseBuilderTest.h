//
// Created by martin on 26/11/2019.
//

#ifndef MLIR_RISEBUILDERTEST_H
#define MLIR_RISEBUILDERTEST_H

#include "mlir/Dialect/Rise/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "mlir/Dialect/Rise/IR/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"

#include "mlir/Dialect/Rise/variant.hpp"

namespace mlir {

namespace rise {
std::unique_ptr<OperationPass<FuncOp>> createRiseBuilderTestPass();
using OutputPathType = mpark::variant<int, Value>;


} // namespace rise
} // namespace mlir

#endif // MLIR_RISEBUILDERTEST_H
