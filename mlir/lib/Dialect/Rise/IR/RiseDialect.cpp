//===- RiseDialect.cpp - Rise IR Dialect registration in MLIR
//---------------===//
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
//
// This file implements the dialect for the Rise IR
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rise/IR/Dialect.h"
#include <iostream>

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace mlir {
namespace rise {

/// Dialect creation
RiseDialect::RiseDialect(mlir::MLIRContext *ctx) : mlir::Dialect("rise", ctx) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Rise/IR/Rise.cpp.inc"
      >();
  ///      Types:                    Nats: Datatypes:
  addTypes<FunType, DataTypeWrapper, Nat, Tuple, ArrayType, ScalarType>();
  addAttributes<DataTypeAttr, NatAttr, LiteralAttr>();
}

/// Parse a type registered to this dialect
mlir::Type RiseDialect::parseType(DialectAsmParser &parser) const {
  if (succeeded(parser.parseOptionalKeyword("fun"))) {
    Type input;
    Type output;
    if (!succeeded(parser.parseLess()))
      return nullptr;
    input = parseType(parser);
    if (!succeeded(parser.parseArrow()))
      return nullptr;
    output = parseType(parser);
    if (!succeeded(parser.parseGreater()))
      return nullptr;
    return FunType::get(parser.getBuilder().getContext(), input, output);
  } else {
    return parseDataType(parser);
  }
}

DataType RiseDialect::parseDataType(DialectAsmParser &parser) const {
  if (succeeded(parser.parseOptionalKeyword("tuple"))) {
    DataType first;
    DataType second;
    if (!succeeded(parser.parseLess()))
      return nullptr;
    first = parseDataType(parser);
    if (!succeeded(parser.parseComma()))
      return nullptr;
    second = parseDataType(parser);
    if (!succeeded(parser.parseGreater()))
      return nullptr;
    return Tuple::get(parser.getBuilder().getContext(), first, second);

  } else if (succeeded(parser.parseOptionalKeyword("array"))) {
    DataType elementType;
    int elementCount;
    if (!succeeded(parser.parseLess()) ||
        !succeeded(parser.parseInteger(elementCount)) ||
        !succeeded(parser.parseComma()))
      return nullptr;
    elementType = parseDataType(parser);
    if (!succeeded(parser.parseGreater()))
      return nullptr;
    return ArrayType::get(
        parser.getBuilder().getContext(),
        Nat::get(parser.getBuilder().getContext(), elementCount), elementType);

  } else if (succeeded(parser.parseOptionalKeyword("scalar"))) {
    Type wrappedType;
    if (!succeeded(parser.parseLess()) ||
        !succeeded(parser.parseType(wrappedType)) ||
        !succeeded(parser.parseGreater()))
      return nullptr;

    return ScalarType::get(parser.getBuilder().getContext(), wrappedType);
  } else
    return nullptr;
}

// This enables printing types inside DataTypeAttrs without "!rise."
void RiseDialect::printTypeInternal(Type type, raw_ostream &stream,
                                    DialectAsmPrinter &printer) const {
  if (ScalarType scalarType = type.dyn_cast<ScalarType>()) {
    stream << "scalar<";
    printTypeInternal(scalarType.getWrappedType(), stream, printer);
    stream << ">";
  } else if (FunType funType = type.dyn_cast<FunType>()) {
    stream << "fun<";
    printTypeInternal(funType.getInput(), stream, printer);
    stream << " -> ";
    printTypeInternal(funType.getOutput(), stream, printer);
    stream << ">";
  } else if (Tuple tuple = type.dyn_cast<Tuple>()) {
    stream << "tuple<";
    printTypeInternal(tuple.getFirst(), stream, printer);
    stream << ", ";
    printTypeInternal(tuple.getSecond(), stream, printer);
    stream << ">";
  } else if (ArrayType array = type.dyn_cast<ArrayType>()) {
    stream << "array<";
    stream << array.getSize().getIntValue();
    stream << ", ";
    printTypeInternal(array.getElementType(), stream, printer);
    stream << ">";
  } else {
    // print type from other dialect
    printer.printType(type);
  }
  return;
}

/// Print a Rise type
void RiseDialect::printType(mlir::Type type, DialectAsmPrinter &printer) const {
  printTypeInternal(type, printer.getStream(), printer);
  return;
}

mlir::Attribute RiseDialect::parseAttribute(DialectAsmParser &parser,
                                            Type type) const {
  // TODO: implement parsing of different literals
  DataType literalType;

  if (succeeded(parser.parseOptionalKeyword("lit"))) {
    if (!succeeded(parser.parseLess()))
      return nullptr;
    double literalValue;
    if (succeeded(parser.parseFloat(literalValue))) {

      if (!succeeded(parser.parseOptionalGreater())) {
        if (!succeeded(parser.parseComma()))
          return nullptr;

        literalType = parseDataType(parser);

        if (!succeeded(parser.parseGreater()))
          return nullptr;

        return LiteralAttr::get(
            getContext(),
            literalType,
            std::to_string(literalValue));
      }
      return LiteralAttr::get(
          getContext(),
          ScalarType::get(getContext(), FloatType::getF32(getContext())),
          std::to_string(literalValue));
    } else {
      return nullptr;
    }
  } else if (succeeded(parser.parseOptionalKeyword("nat"))) {
    if (!succeeded(parser.parseLess()))
      return nullptr;
    int natValue;
    if (!succeeded(parser.parseInteger(natValue)) ||
        !succeeded(parser.parseGreater()))
      return nullptr;
    return NatAttr::get(getContext(), Nat::get(getContext(), natValue));
  } else {
    // we have a DataType Attribute
    DataType type = parseDataType(parser);
    return DataTypeAttr::get(getContext(), type);
  }

  return nullptr;
}

void RiseDialect::printAttribute(Attribute attribute,
                                 DialectAsmPrinter &printer) const {
  raw_ostream &os = printer.getStream();
  if (DataTypeAttr dataTypeAttr = attribute.dyn_cast<DataTypeAttr>()) {
    printTypeInternal(dataTypeAttr.getValue(), printer.getStream(), printer);
  } else if (NatAttr natAttr = attribute.dyn_cast<NatAttr>()) {
    printer << "nat<";
    printer << natAttr.getValue().getIntValue();
    printer << ">";
  } else if (LiteralAttr literalAttr = attribute.dyn_cast<LiteralAttr>()) {
    printer << "lit<";
    printer << literalAttr.getValue() << ", ";
    printTypeInternal(literalAttr.getType(), printer.getStream(), printer);
    printer << ">";
  } else {
    os << "unknown attribute";
  }
  return;
}

} // end namespace rise
} // end namespace mlir