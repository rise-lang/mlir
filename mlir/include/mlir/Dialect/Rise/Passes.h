//
// Created by martin on 26/11/2019.
//

#ifndef MLIR_DIALECT_RISE_PASSES_H
#define MLIR_DIALECT_RISE_PASSES_H

#include "mlir/Support/LLVM.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
class AffineDialect;

void populateRiseToStdConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateRiseToImpConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);

#define GEN_PASS_CLASSES
#include "mlir/Dialect/Rise/Passes.h.inc"

namespace rise {

std::unique_ptr<OperationPass<FuncOp>> createConvertRiseToImperativePass();

} // namespace rise
} // namespace mlir

#endif // MLIR_DIALECT_RISE_PASSES_H
