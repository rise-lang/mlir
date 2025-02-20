//
// Created by martin on 26/11/2019.
//

#ifndef MLIR_DIALECT_RISE_PASSES_H
#define MLIR_DIALECT_RISE_PASSES_H

#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
class AffineDialect;
namespace scf {
class SCFDialect;
}

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
