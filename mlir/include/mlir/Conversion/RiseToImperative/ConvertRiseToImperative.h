//
// Created by martin on 26/11/2019.
//

#ifndef MLIR_CONVERTRISETOIMPERATIVE_H
#define MLIR_CONVERTRISETOIMPERATIVE_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Rise/Types.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Dialect/Rise/Dialect.h"
#include "mlir/Dialect/Rise/Passes.h"

namespace mlir{
    namespace rise {
        Block::OpListType* AccT(Block *expression, mlir::Value output, PatternRewriter &rewriter);
        Block::OpListType* ConT(Block *expression, mlir::Value output, PatternRewriter &rewriter);
        }
}


#endif //MLIR_CONVERTRISETOIMPERATIVE_H
