//
// Created by martin on 29/10/2020.
//

#ifndef LLVM_ELEVATEREWRITEDRIVER_H
#define LLVM_ELEVATEREWRITEDRIVER_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Elevate2/core.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"


namespace mlir {

// or better ElevateStrategyApplicator?
class ElevateRewriteDriver : public PatternRewriter {
public:
  explicit ElevateRewriteDriver(MLIRContext *ctx);

//  mlir::elevate2::RewriteResult rewrite(Operation *op,
//                                        mlir::elevate2::Strategy strategy);

  // These are hooks implemented for PatternRewriter.
protected:
  // Implement the hook for inserting operations, and make sure that newly
  // inserted ops are added to the worklist for processing.
  void notifyOperationInserted(Operation *op) override;

  // If an operation is about to be removed, make sure it is not in our
  // worklist anymore because we'd get dangling references to it.
  void notifyOperationRemoved(Operation *op) override;

  // When the root of a pattern is about to be replaced, it can trigger
  // simplifications to its users - make sure to add them to the worklist
  // before the root is changed.
  void notifyRootReplaced(Operation *op) override;
};
}

#endif // LLVM_ELEVATEREWRITEDRIVER_H
