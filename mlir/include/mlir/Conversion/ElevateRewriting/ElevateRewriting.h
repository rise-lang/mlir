//
// Created by martin on 26/11/2019.
//

#ifndef MLIR_ELEVATEREWRITING_H
#define MLIR_ELEVATEREWRITING_H

//#include "mlir/Elevate/ElevateRewriter.h"
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

//#include "mlir/Dialect/Rise/Elevate/traversal.h"
//#include "mlir/Dialect/Rise/Elevate/algorithmic.h"

#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"

#include "mlir/Dialect/Rise/variant.hpp"

namespace mlir {

namespace rise {
std::unique_ptr<OperationPass<FuncOp>> createElevateRewritingPass();

using OutputPathType = mpark::variant<int, Value>;


} // namespace rise
} // namespace mlir

#endif // MLIR_ELEVATEREWRITING_H
