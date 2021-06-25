//===-- Passes.td - Rise pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_RISE_PASSES
#define MLIR_DIALECT_RISE_PASSES

include "mlir/Pass/PassBase.td"

def RiseBuilderTest
    : FunctionPass<"rise-builder-test"> {
  let summary = "Test Rise MLIR builders";
  let constructor = "mlir::rise::createRiseBuilderTestPass()";
  let dependentDialects = ["mlir::rise::RiseDialect", "scf::SCFDialect", "AffineDialect"];
}

def RiseToImperative
    : FunctionPass<"convert-rise-to-imperative"> {
  let summary = "Lower all functional primitives of the rise dialect to imperative";
  let constructor = "mlir::rise::createConvertRiseToImperativePass()";
  let dependentDialects = ["mlir::rise::RiseDialect", "scf::SCFDialect", "AffineDialect"];
}

def ElevateRewriting
    : FunctionPass<"elevate-rewriting"> {
  let summary = "";
  let constructor = "mlir::rise::createElevateRewritingPass()";
  let dependentDialects = ["mlir::rise::RiseDialect", "scf::SCFDialect", "AffineDialect"];
}

#endif // MLIR_DIALECT_RISE_PASSES
