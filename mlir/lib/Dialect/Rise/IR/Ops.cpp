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

#include "mlir/Dialect/Rise/IR/Ops.h"
#include <iostream>

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"

////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////

using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace mlir {
namespace rise {

//===----------------------------------------------------------------------===//
// RiseModuleOp
//===----------------------------------------------------------------------===//
ParseResult parseRiseModuleOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  SmallVector<OpAsmParser::OperandType, 4> arguments;
  SmallVector<Type, 4> argumentTypes = SmallVector<Type, 4>();

  /// Indicate, that this is not lowered
  // TODO: This is temporary
  result.addAttribute("lowered", BoolAttr::get(false, builder.getContext()));

  // Parse body of lambda
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, arguments, argumentTypes))
    return failure();

  LambdaOp::ensureTerminator(*body, builder, result.location);
  return success();
}

//===----------------------------------------------------------------------===//
// RiseModuleOp
//===----------------------------------------------------------------------===//
ParseResult parseRiseContinuationTranslation(OpAsmParser &parser,
    OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType contValue;

  if (parser.parseOperand(contValue))
    return failure();

  if (parser.resolveOperandUnsafe(contValue, result.operands))
    return failure();

  /// Maybe add the ability to request a result type as in "This will be
  // translated into a memref


  return success();
}



//===----------------------------------------------------------------------===//
// LambdaOp
//===----------------------------------------------------------------------===//
ParseResult parseLambdaOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  SmallVector<OpAsmParser::OperandType, 4> arguments;
  SmallVector<Type, 4> argumentTypes = SmallVector<Type, 4>();
  FunType funType;

  // arguments for the lambda
  if (parser.parseRegionArgumentList(arguments, OpAsmParser::Delimiter::Paren))
    failure();

  // result type of this lambda
  if (parser.parseColon() || parser.parseType(funType))
    return failure();

  // arguments have to fit the given lambda type
  argumentTypes.push_back(funType.getInput());
  for (int i = 1; i < arguments.size(); i++) {
    if (funType.getOutput().isa<FunType>()) {
      funType = funType.getOutput().dyn_cast<FunType>();
      argumentTypes.push_back(funType.getInput());
    } else {
      parser.emitError(parser.getCurrentLocation())
          << ": number of arguments: " << std::to_string(i)
          << " is too high for specified funType";
      return failure();
    }
  }

  // Parse body of lambda
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, arguments, argumentTypes))
    return failure();

  LambdaOp::ensureTerminator(*body, builder, result.location);
  result.addTypes(funType);
  return success();
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//
ParseResult parseApplyOp(OpAsmParser &parser, OperationState &result) {
  bool simplified = true;

  auto &builder = parser.getBuilder();

  OpAsmParser::OperandType funOperand;
  FunType funType;
  SmallVector<OpAsmParser::OperandType, 4> arguments;
  SmallVector<Type, 4> argumentTypes = SmallVector<Type, 4>();

  // parse function
  if (parser.parseOperand(funOperand))
    return failure();

  if (!simplified) {
    // parse type of the function
    if (parser.parseColonType(funType))
      return failure();
  }

  // TODO: we dont want to have to explicitly give our type
  /// resolve operand adds it to the operands of this operation.
  /// I have not found another way to add it, yet
  /// result.addOperands expects a mlir::Value, which has to contain the Type of
  /// the Operand already, which I don't know here
  if (!simplified) {
    if (parser.resolveOperand(funOperand, funType, result.operands))
      failure();
  } else {
    if (parser.resolveOperandUnsafe(funOperand, result.operands))
      failure();
  }
  funType = result.operands.front().getType().dyn_cast<FunType>();

  // parse arguments
  if (parser.parseTrailingOperandList(arguments))
    failure();

  // get types of arguments from the function type and determine
  // the result type of this apply operation
  argumentTypes.push_back(funType.getInput());
  for (int i = 1; i < arguments.size(); i++) {
    if (funType.getOutput().isa<FunType>()) {
      funType = funType.getOutput().dyn_cast<FunType>();
      argumentTypes.push_back(funType.getInput());
    } else {
      parser.emitError(parser.getCurrentLocation())
          << "expected a maximum " << std::to_string(i)
          << " arguments for this function.";
      return failure();
    }
  }
  if (parser.resolveOperands(arguments, argumentTypes,
                             parser.getCurrentLocation(), result.operands))
    failure();

  result.addTypes(funType.getOutput());
  result.setOperandListToResizable(true);
  return success();
}

//===----------------------------------------------------------------------===//
// ParseLiteralOp
//===----------------------------------------------------------------------===//
/// This format is not the one used in the paper and will change to it soon.
/// current Format:
///         rise.literal #rise.int<42>
///         rise.literal #rise.array<2, rise.int, [1,2]>
///         rise.literal #rise.array<2.3, !rise.int, [[1,2,3],[4,5,6]]>
// TODO: restructure the literal attribute to clearly differ between type and
// value
ParseResult parseLiteralOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  LiteralAttr attr;

  // type and value of literal
  if (parser.parseAttribute(attr, "literal", result.attributes))
    return failure();

  std::cout << "\nstill alive in parseLiteral\n";

  result.addTypes(DataTypeWrapper::get(builder.getContext(), attr.getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// Map
//===----------------------------------------------------------------------===//

/// map: {n : nat} → {s t : data} → (s → t ) → n.s → n.t
ParseResult parseMapOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  NatAttr n;
  DataTypeAttr s, t;
  result.setOperandListToResizable();

  // length of array
  if (parser.parseAttribute(n, "n", result.attributes))
    failure();

  // input array element type
  if (parser.parseAttribute(s, "s", result.attributes))
    failure();

  // output array element type
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(FunType::get(
      builder.getContext(),
      FunType::get(builder.getContext(),
                   DataTypeWrapper::get(builder.getContext(), s.getValue()),
                   DataTypeWrapper::get(builder.getContext(), t.getValue())),
      FunType::get(
          builder.getContext(),
          DataTypeWrapper::get(
              builder.getContext(),
              ArrayType::get(builder.getContext(), n.getValue(), s.getValue())),
          DataTypeWrapper::get(builder.getContext(),
                               ArrayType::get(builder.getContext(),
                                              n.getValue(), t.getValue())))));
  return success();
}

//===----------------------------------------------------------------------===//
// Reduce
//===----------------------------------------------------------------------===//

/// reduce: {n : nat} → {s t : data} → (s → t → t ) → t → n.s → t
ParseResult parseReduceOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  NatAttr n;
  DataTypeAttr s, t;
  result.setOperandListToResizable();

  // number of elements in Array
  if (parser.parseAttribute(n, "n", result.attributes))
    failure();

  // elementType of Array
  if (parser.parseAttribute(s, "s", result.attributes))
    failure();

  // accumulator type
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(FunType::get(
      builder.getContext(),
      FunType::get(
          builder.getContext(),
          DataTypeWrapper::get(builder.getContext(), s.getValue()),
          FunType::get(
              builder.getContext(),
              DataTypeWrapper::get(builder.getContext(), t.getValue()),
              DataTypeWrapper::get(builder.getContext(), t.getValue()))),
      FunType::get(
          builder.getContext(),
          DataTypeWrapper::get(builder.getContext(), t.getValue()),
          FunType::get(
              builder.getContext(),
              DataTypeWrapper::get(builder.getContext(),
                                   ArrayType::get(builder.getContext(),
                                                  n.getValue(), s.getValue())),
              DataTypeWrapper::get(builder.getContext(), t.getValue())))));

  return success();
}

//===----------------------------------------------------------------------===//
// Tuple Ops
//===----------------------------------------------------------------------===//

/// zip: {n : nat} → {s t : data} → n.s → n.t → n.(s × t )
ParseResult parseZipOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  NatAttr n;
  DataTypeAttr s, t;

  // number of elements in Array
  if (parser.parseAttribute(n, "n", result.attributes))
    return failure();

  // elementType of first Array
  if (parser.parseAttribute(s, "s", result.attributes))
    return failure();

  // elementType of second Array
  if (parser.parseAttribute(t, "t", result.attributes))
    return failure();

  result.addTypes(FunType::get(
      builder.getContext(),
      DataTypeWrapper::get(
          builder.getContext(),
          ArrayType::get(builder.getContext(), n.getValue(), s.getValue())),
      FunType::get(
          builder.getContext(),
          DataTypeWrapper::get(
              builder.getContext(),
              ArrayType::get(builder.getContext(), n.getValue(), t.getValue())),
          DataTypeWrapper::get(
              builder.getContext(),
              ArrayType::get(builder.getContext(), n.getValue(),
                             Tuple::get(builder.getContext(), s.getValue(),
                                        t.getValue()))))));
  return success();
}

/// tuple: {s t : data} → s → t → s × t
ParseResult parseTupleOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  DataTypeAttr s, t;
  result.setOperandListToResizable();

  // type of first element
  if (parser.parseAttribute(s, "s", result.attributes))
    failure();

  // type of second element
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(FunType::get(
      builder.getContext(),
      DataTypeWrapper::get(builder.getContext(), s.getValue()),
      FunType::get(
          builder.getContext(),
          DataTypeWrapper::get(builder.getContext(), t.getValue()),
          DataTypeWrapper::get(
              builder.getContext(),
              Tuple::get(builder.getContext(), s.getValue(), t.getValue())))));
  return success();
}

/// fst: {s t : data} → s × t → s
ParseResult parseFstOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  DataTypeAttr s, t;
  result.setOperandListToResizable();

  // type of first element
  if (parser.parseAttribute(s, "s", result.attributes))
    failure();

  // type of second element
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(
      FunType::get(builder.getContext(),
                   DataTypeWrapper::get(builder.getContext(),
                                        Tuple::get(builder.getContext(),
                                                   s.getValue(), t.getValue())),
                   DataTypeWrapper::get(builder.getContext(), s.getValue())));
  return success();
}

/// snd: {s t : data} → s × t → t
ParseResult parseSndOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  DataTypeAttr s, t;
  result.setOperandListToResizable();

  // type of first element
  if (parser.parseAttribute(s, "s", result.attributes))
    failure();

  // type of second element
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(
      FunType::get(builder.getContext(),
                   DataTypeWrapper::get(builder.getContext(),
                                        Tuple::get(builder.getContext(),
                                                   s.getValue(), t.getValue())),
                   DataTypeWrapper::get(builder.getContext(), t.getValue())));
  return success();
}

//===----------------------------------------------------------------------===//
// Arithmetics
//===----------------------------------------------------------------------===//

/// add: {t : data} → t → t → t
ParseResult parseAddOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  DataTypeAttr t;

  // type of summands
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(FunType::get(
      builder.getContext(),
      DataTypeWrapper::get(builder.getContext(), t.getValue()),
      FunType::get(builder.getContext(),
                   DataTypeWrapper::get(builder.getContext(), t.getValue()),
                   DataTypeWrapper::get(builder.getContext(), t.getValue()))));
  return success();
}

/// mult: {t : data} → t → t → t
ParseResult parseMultOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  DataTypeAttr t;

  // type of factors
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(FunType::get(
      builder.getContext(),
      DataTypeWrapper::get(builder.getContext(), t.getValue()),
      FunType::get(builder.getContext(),
                   DataTypeWrapper::get(builder.getContext(), t.getValue()),
                   DataTypeWrapper::get(builder.getContext(), t.getValue()))));
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//
ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::OperandType value;
  Type type;
  result.setOperandListToResizable();

  // return value
  if (parser.parseOperand(value))
    failure();

  // type of return value
  // TODO: we do not want to have to give this explicitly
  if (parser.parseColonType(type))
    failure();

  if (parser.resolveOperand(value, type, result.operands))
    failure();
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Rise/IR/Rise.cpp.inc"
} // end namespace rise
} // end namespace mlir
