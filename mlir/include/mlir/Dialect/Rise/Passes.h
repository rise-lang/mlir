//
// Created by martin on 26/11/2019.
//

#ifndef MLIR_DIALECT_RISE_PASSES_H
#define MLIR_DIALECT_RISE_PASSES_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class ModuleOp;
template <typename T> class OpPassBase;


// Owning list of rewriting patterns.
class OwningRewritePatternList;

///
void populateRiseToStdConversionPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx);

namespace rise {




std::unique_ptr<OpPassBase<ModuleOp>> createConvertRiseToStandardPass();




} //namespace rise
} //namespace mlir



#endif //MLIR_DIALECT_RISE_PASSES_H
