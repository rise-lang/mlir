//
// Created by martin on 10/3/20.
//

#ifndef LLVM_ELEVATEREWRITER_H
#define LLVM_ELEVATEREWRITER_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

class ElevateRewriter
{
public:
  static ElevateRewriter& getInstance();
  mlir::PatternRewriter *rewriter;

  ElevateRewriter(ElevateRewriter const&) = delete;
  void operator=(ElevateRewriter const&) = delete;

private:
  ElevateRewriter() {}
};

ElevateRewriter& ElevateRewriter::getInstance() {
  static ElevateRewriter instance;
  return instance;
}

#endif // LLVM_ELEVATEREWRITER_H
