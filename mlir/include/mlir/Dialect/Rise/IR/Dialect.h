//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef MLIR_RISE_DIALECT_H_
#define MLIR_RISE_DIALECT_H_

#include "mlir/IR/Dialect.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser.h"

#include "mlir/Dialect/Rise/IR/Ops.h"
#include "mlir/Dialect/Rise/IR/Types.h"

namespace mlir {
class Builder;

namespace rise {


/// This is the definition of the Rise dialect.
class RiseDialect : public mlir::Dialect {
public:
  explicit RiseDialect(mlir::MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "rise"; }

  /// Hook for custom parsing of types
  mlir::Type parseType(DialectAsmParser &parser) const override;
  DataType parseDataType(DialectAsmParser &parser) const;

  /// Hook for custom printing of types
  void printType(mlir::Type type, DialectAsmPrinter &) const override;
  void printTypeInternal(mlir::Type type, raw_ostream &stream, DialectAsmPrinter &printer) const;

  /// Hook for custom parsing of Attributes
  mlir::Attribute parseAttribute(DialectAsmParser &parser,
                                 Type type) const override;

  /// Hook for custom printing of Attributes
  void printAttribute(Attribute, DialectAsmPrinter &) const override;

  static void dumpRiseExpression(Operation *op);
  static void dumpRiseExpression2(LoweringUnitOp* op);
  static DataType getAsDataType(Type type);
  static DataType getFunTypeOutput(FunType funType);
  void initialize();
  friend class ::mlir::MLIRContext;
};

} // end namespace rise
} // end namespace mlir
#endif // MLIR_RISE_DIALECT_H_
