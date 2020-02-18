//
// Created by martin on 26/11/2019.
//

#ifndef MLIR_DIALECT_RISE_PASSES_H
#define MLIR_DIALECT_RISE_PASSES_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class ModuleOp;
template <typename T> class OpPassBase;
class OwningRewritePatternList;

    std::unique_ptr<OpPassBase<ModuleOp>> createConvertRiseToStandardPass();
    std::unique_ptr<OpPassBase<ModuleOp>> createConvertRiseToImperativePass();

void populateRiseToStdConversionPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx);
void populateRiseToImpConversionPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx);




//} //namespace rise
} //namespace mlir



#endif //MLIR_DIALECT_RISE_PASSES_H
