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

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
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
  ///      Types:                              Nats:               Datatypes:
  addTypes<FunType, DataTypeWrapper, Nat, Int, Float, Tuple, ArrayType>();
  addAttributes<DataTypeAttr, NatAttr, LiteralAttr>();
}

/// Parse a type registered to this dialect
mlir::Type RiseDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeString = parser.getFullSymbolSpec();
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());

  if (typeString.startswith("!rise."))
    typeString.consume_front("!rise.");

  if (typeString.startswith("fun") || typeString.startswith("data")) {
    return parseRiseType(typeString, loc);
  }
  if (typeString.startswith("array") || typeString.startswith("tuple") ||
      typeString.startswith("int") || typeString.startswith("float")) {
    return parseDataType(typeString, loc);
  }
  if (typeString.startswith("nat")) {
    return parseNat(typeString, loc);
  }
  emitError(loc, "Invalid Rise type '" + typeString + "'");
  return nullptr;
}

RiseType RiseDialect::parseRiseType(StringRef typeString,
                                    mlir::Location loc) const {

  if (typeString.startswith("fun")) {
    if (!typeString.consume_front("fun<") || !typeString.consume_back(">")) {
      emitError(loc, "rise.fun delimiter <...> mismatch");
      return nullptr;
    }

    /// As FunTypes may have other FunTypes as input and output, we have to find
    /// the "->" which belongs to this FunType:
    std::string functionString = typeString.str();
    std::string subString;
    size_t pos, oldPos = 0;
    pos = functionString.find("->");
    subString = functionString.substr(0, pos);

    while (subString.find("fun") != SIZE_MAX) {
      oldPos = pos;
      pos = functionString.find("->", pos + 2); // find next occurence
      // If not found
      if (pos == SIZE_MAX) {
        pos = oldPos;
        break;
      }
      subString = functionString.substr(oldPos + 2, pos - (oldPos + 2));
    }

    ///  Split into input type and output type and parse them
    StringRef inputDataString, outputDataString;
    inputDataString = typeString.substr(0, pos);
    outputDataString = typeString.substr(pos + 2, typeString.npos);
    if (outputDataString.empty()) {
      emitError(loc, "expected -> to separate input type and output type '")
          << typeString << "'";
      return nullptr;
    }
    inputDataString = inputDataString.trim();
    outputDataString = outputDataString.trim();

    /// This is for prettier printing:
    /// We can specify "!rise.type" as well as just "type" inside a FunType
    if (inputDataString.startswith("!rise."))
      inputDataString.consume_front("!rise.");
    if (outputDataString.startswith("!rise."))
      outputDataString.consume_front("!rise.");

    RiseType inputData = parseRiseType(inputDataString, loc);
    RiseType outputData = parseRiseType(outputDataString, loc);

    return FunType::get(getContext(), inputData, outputData);
  }
  if (typeString.startswith("data")) {
    if (!typeString.consume_front("data<") || !typeString.consume_back(">")) {

      emitError(loc, "rise.data delimiter <...> mismatch")
          << " string: " << typeString.str();
      return nullptr;
    }
    DataType wrappedType = parseDataType(typeString, loc);

    return DataTypeWrapper::get(getContext(), wrappedType);
  }
  if (typeString.equals("int")) {
    return DataTypeWrapper::get(getContext(), Int::get(getContext()));
  }
  if (typeString.equals("float")) {
    return DataTypeWrapper::get(getContext(), Float::get(getContext()));
  }
  emitError(loc, "parsing of Rise type failed.");
  return nullptr;
}

DataType RiseDialect::parseDataType(StringRef typeString,
                                    mlir::Location loc) const {
  if (typeString.startswith("!rise."))
    typeString.consume_front("!rise.");

  if (typeString.startswith("array") || typeString.startswith("!rise.array")) {
    if (!typeString.consume_front("array<") || !typeString.consume_back(">")) {
      emitError(loc, "rise.array delimiter <...> mismatch");
      return nullptr;
    }

    /// Split into size and elementType at the first `,`
    StringRef sizeString, elementTypeString;
    std::tie(sizeString, elementTypeString) = typeString.split(',');
    if (elementTypeString.empty()) {
      emitError(loc, "expected comma to separate size and elementType'")
          << typeString << "'";
      return nullptr;
    }

    /// getting rid of leading or trailing whitspaces etc.
    sizeString = sizeString.trim();
    elementTypeString = elementTypeString.trim();

    int size = std::stoi(sizeString.str());
    Nat natSize = Nat::get(getContext(), size);

    if (elementTypeString.startswith("!rise."))
      elementTypeString.consume_front("!rise.");
    DataType elementType = parseDataType(elementTypeString, loc);

    return ArrayType::get(getContext(), natSize, elementType);
  }
  if (typeString.startswith("tuple<")) {
    if (!typeString.consume_front("tuple<") || !typeString.consume_back(">")) {
      emitError(loc, "rise.tuple delimiter <...> mismatch");
      return nullptr;
    }
    StringRef leftTypeString, rightTypeString;
    std::tie(leftTypeString, rightTypeString) = typeString.split(',');
    if (rightTypeString.empty()) {
      emitError(loc, "expected comma to separate left and right Type in tuple'")
          << typeString << "'";
      return nullptr;
    }

    leftTypeString = leftTypeString.trim();
    rightTypeString = rightTypeString.trim();

    DataType leftType = parseDataType(leftTypeString, loc);
    DataType rightType = parseDataType(rightTypeString, loc);

    return Tuple::get(getContext(), leftType, rightType);
  }
  if (typeString.startswith("float")) {
    return Float::get(getContext());
  }

  if (typeString.startswith("int")) {
    return Int::get(getContext());
  }
  emitError(loc, "parsing of Rise DataType failed.");
  return nullptr;
}

Nat RiseDialect::parseNat(StringRef typeString, mlir::Location loc) const {
  if (typeString.startswith("nat")) {
    if (!typeString.consume_front("<") || !typeString.consume_back(">")) {
      emitError(loc, "rise.nat delimiter <...> mismatch");
      return nullptr;
    }
    int size = std::stoi(typeString.str());
    return Nat::get(getContext(), size);
  }
  emitError(loc, "parsing of Rise nat failed.");
  return nullptr;
}

std::string static stringForType(Type type) {
  switch (type.getKind()) {
  default: {
    return "unknown rise type";
  }
  case RiseTypeKind::RISE_FLOAT: {
    return "float";
  }
  case RiseTypeKind::RISE_INT: {
    return "int";
  }
  case RiseTypeKind::RISE_NAT: {
    return "nat";
  }
  case RiseTypeKind::RISE_TUPLE: {
    return "tuple<" + stringForType(type.dyn_cast<Tuple>().getFirst()) + ", " +
           stringForType(type.dyn_cast<Tuple>().getSecond()) + ">";
  }
  case RiseTypeKind::RISE_ARRAY: {
    return "array<" +
           std::to_string(type.dyn_cast<ArrayType>().getSize().getIntValue()) +
           ", " + stringForType(type.dyn_cast<ArrayType>().getElementType()) +
           ">";
  }
  case RiseTypeKind::RISE_FUNTYPE: {
    return "fun<" + stringForType(type.dyn_cast<FunType>().getInput()) +
           " -> " + stringForType(type.dyn_cast<FunType>().getOutput()) + ">";
  }
  case RiseTypeKind::RISE_DATATYPE_WRAPPER: {
    return "data<" +
           stringForType(type.dyn_cast<DataTypeWrapper>().getDataType()) + ">";
  }
  }
}

/// Print a Rise type
void RiseDialect::printType(mlir::Type type, DialectAsmPrinter &printer) const {
  raw_ostream &os = printer.getStream();
  os << stringForType(type);
}

///         rise.literal #rise.int<42>
///         rise.literal #rise.array<2, rise.int, [1,2]>
///         rise.literal #rise.array<2.3, !rise.int, [[1,2,3],[4,5,6]]>
mlir::Attribute RiseDialect::parseAttribute(DialectAsmParser &parser,
                                            Type type) const {
  StringRef attrString = parser.getFullSymbolSpec();
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  // we only have RiseTypeAttribute
  if (attrString.startswith("lit<")) {
    return parseLiteralAttribute(attrString, loc);
  }
  if (attrString.startswith("array") || attrString.startswith("tuple") ||
      attrString.startswith("int") || attrString.startswith("float")) {
    return parseDataTypeAttribute(attrString, loc);
  }
  if (attrString.startswith("nat")) {
    return parseNatAttribute(attrString, loc);
  }
  emitError(loc, "Invalid Rise attribute '" + attrString + "'");
  return nullptr;
}

DataTypeAttr RiseDialect::parseDataTypeAttribute(StringRef attrString,
                                                 mlir::Location loc) const {
  if (attrString.equals("int"))
    return DataTypeAttr::get(getContext(), Int::get(getContext()));
  if (attrString.equals("float"))
    return DataTypeAttr::get(getContext(), Float::get(getContext()));
  if (attrString.startswith("tuple<"))
    return DataTypeAttr::get(getContext(), parseDataType(attrString, loc));
  if (attrString.startswith("array<"))
    return DataTypeAttr::get(getContext(), parseDataType(attrString, loc));
  return nullptr;
}

NatAttr RiseDialect::parseNatAttribute(StringRef attrString,
                                       mlir::Location loc) const {
  if (!attrString.consume_front("nat<") || !attrString.consume_back(">")) {
    emitError(loc, "#rise.nat delimiter <...> mismatch");
    return nullptr;
  }
  int natValue = std::stoi(attrString.str());
  return NatAttr::get(getContext(), Nat::get(getContext(), natValue));
}

/// recursive utiliy function to determine the structure of an Array from a
/// string
DataType static getArrayStructure(mlir::MLIRContext *context,
                                  StringRef structureString,
                                  DataType elementType, mlir::Location loc) {

  StringRef currentDim, restStructure;
  std::tie(currentDim, restStructure) = structureString.split('.');

  if (restStructure == "") {
    return ArrayType::get(
        context, Nat::get(context, std::stoi(currentDim.str())), elementType);
  } else {
    return ArrayType::get(
        context, Nat::get(context, std::stoi(currentDim.str())),
        getArrayStructure(context, restStructure, elementType, loc));
  }
}

/// This version still uses a syntax which couples type and value tightly
/// New proposed structure: separate type from the literal value more strictly.
///         rise.literal #rise.lit<int, 42>
///         rise.literal #rise.lit<array<2, array<2, int>>, [[1,2],[3,4]]
LiteralAttr RiseDialect::parseLiteralAttribute(StringRef attrString,
                                               mlir::Location loc) const {

  if (!attrString.consume_front("lit<") || !attrString.consume_back(">")) {
    emitError(loc, "#rise.lit delimiter <...> mismatch");
    return nullptr;
  }
  /// format:
  ///     #rise.lit<int<int_value>>
  ///     #rise.lit<int<42>>
  if (attrString.startswith("int")) {
    if (!attrString.consume_front("int<") || !attrString.consume_back(">")) {
      emitError(loc, "#rise.int delimiter <...> mismatch");
      return nullptr;
    }
    return LiteralAttr::get(getContext(), Int::get(getContext()),
                            attrString.str());
  }
  if (attrString.startswith("float")) {
    if (!attrString.consume_front("float<") || !attrString.consume_back(">")) {
      emitError(loc, "#rise.float delimiter <...> mismatch");
      return nullptr;
    }
    return LiteralAttr::get(getContext(), Float::get(getContext()),
                            attrString.str());
  }
  /// format
  ///     #rise.lit<array<array_structure, element_type, values>
  ///     #rise.lit<array<2, !rise.int, [1,2]>>
  ///     #rise.lit<array<2.3, !rise.int, [[1,2,3],[4,5,6]]>>
  if (attrString.startswith("array")) {
    if (!attrString.consume_front("array<") || !attrString.consume_back(">")) {
      emitError(loc, "#rise.array delimiter <...> mismatch");
      return nullptr;
    }
    StringRef structureString, elementTypeString, valueString;
    std::tie(structureString, elementTypeString) = attrString.split(',');
    std::tie(elementTypeString, valueString) = elementTypeString.split(',');

    if (valueString.empty()) {
      emitError(loc, "expected commas to separate structure, elementType and "
                     "values of an array,"
                     " ex: rise.array<2, !rise.int, [1,2]>'")
          << attrString << "'";
      return nullptr;
    }

    // getting rid of leading or trailing whitspaces etc.
    structureString = structureString.trim();
    elementTypeString = elementTypeString.trim();
    valueString = valueString.trim();

    DataType elementType = RiseDialect::parseDataType(elementTypeString, loc);

    return LiteralAttr::get(
        getContext(),
        getArrayStructure(getContext(), structureString, elementType, loc),
        valueString.str());
  }
  emitError(loc, "parsing of LiteralAttr failed");
  return nullptr;
}

std::string static stringForAttribute(Attribute attribute) {
  switch (attribute.getKind()) {
  default: {
    return "unknown rise type";
  }
  case RiseAttributeKind::DATATYPE_ATTR: {
    return "data<" +
           stringForType(attribute.dyn_cast<DataTypeAttr>().getValue()) + ">";
  }
  case RiseAttributeKind::NAT_ATTR: {
    return "nat<" +
           std::to_string(
               attribute.dyn_cast<NatAttr>().getValue().getIntValue()) +
           ">";
  }
  case RiseAttributeKind::LITERAL_ATTR: {
    switch (attribute.dyn_cast<LiteralAttr>().getType().getKind()) {
    default: {
      return "unknown rise type";
    }
    case RiseTypeKind::RISE_INT: {
      return "lit<int<" + attribute.dyn_cast<LiteralAttr>().getValue() + ">>";
    }
    case RiseTypeKind::RISE_FLOAT: {
      return "lit<float>";
    }
    case RiseTypeKind::RISE_NAT: {
      return "lat<nat>";
    }
    case RiseTypeKind::RISE_ARRAY: {
      return "lit<array<" + attribute.dyn_cast<LiteralAttr>().getValue() + ">>";
    }
    }
  }
  }
}

void RiseDialect::printAttribute(Attribute attribute,
                                 DialectAsmPrinter &printer) const {
  raw_ostream &os = printer.getStream();
  os << stringForAttribute(attribute);
  return;
}

} // end namespace rise
} // end namespace mlir