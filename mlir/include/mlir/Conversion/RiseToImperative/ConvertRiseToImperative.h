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
// using std::variant would be better, but that is C++17
// TODO: This is not complete. We have to be able to represent i%m etc.
//       Also representing fst and snd as bool is just a placeholder
union OutputPathElement {
  int idx;
  bool fst;
  Value identifier;
};
// std::variant would be nice to have here. Look at how this works. I need a sum
// type for my path, I think
using OutputPathType = llvm::SmallVector<OutputPathElement, 10>;
mlir::Value AccT(ApplyOp apply, Value out,
                 PatternRewriter &rewriter);
mlir::Value ConT(mlir::Value contValue, Block::iterator contLocation,
                 PatternRewriter &rewriter);
void Substitute(LambdaOp lambda, llvm::SmallVector<Value, 10> args);
LambdaOp expandToLambda(mlir::Value value, PatternRewriter &rewriter);

} // namespace rise
} // namespace mlir

#endif // MLIR_CONVERTRISETOIMPERATIVE_H
