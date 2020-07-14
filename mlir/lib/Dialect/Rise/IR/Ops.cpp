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
#include "mlir/Dialect/Rise/EDSC/Builders.h"

#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
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

using namespace mlir::edsc::type;

namespace mlir {
namespace rise {

//===----------------------------------------------------------------------===//
// RiseEmbedOp
//===----------------------------------------------------------------------===//
LogicalResult parseEmbedOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  SmallVector<OpAsmParser::OperandType, 4> operands;
  SmallVector<Type, 4> argumentTypes;
  SmallVector<Type, 4> argumentTypesUnpacked;

  if (parser.parseLParen())
    return failure();

  if (parser.parseOperandList(operands))
    return failure();

  if (parser.parseRParen())
    return failure();

  for (auto operand : operands)
    parser.resolveOperand(operand, result.operands);

  for (auto val : result.operands) {
    argumentTypes.push_back(val.getType());
  }

  for (auto type : argumentTypes) {
    ScalarType scalar = type.dyn_cast<ScalarType>();
    argumentTypesUnpacked.push_back(scalar.getWrappedType());
  }

  // Parse body of embed
  Region *body = result.addRegion();

  if (parser.parseRegion(*body, operands, argumentTypesUnpacked, true))
    return failure();

  result.addTypes(ScalarType::get(builder.getContext(),
                                  FloatType::getF32(builder.getContext())));
  return success();
}

void EmbedOp::build(
    OpBuilder &builder, OperationState &result, Type wrapped,
    ValueRange exposedValues,
    function_ref<Value(OpBuilder &, Location, MutableArrayRef<BlockArgument>)>
        bodyBuilder) {
  result.addTypes(wrapped);
  result.addOperands(exposedValues);

  Region *embedRegion = result.addRegion();
  Block *body = new Block();

  for (Value val : exposedValues) {
    assert(val.getType().isa<ScalarType>() &&
           "Only scalar Types can be exposed with rise.embed!");
    body->addArgument(val.getType().dyn_cast<ScalarType>().getWrappedType());
  }
  embedRegion->push_back(body);
  if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);
    Value returnValue =
        bodyBuilder(builder, result.location, body->getArguments());
    builder.create<rise::ReturnOp>(returnValue.getLoc(),
                                   ValueRange{returnValue});
  } else {
    builder.create<rise::ReturnOp>(result.location, ValueRange{});
  }
}

//===----------------------------------------------------------------------===//
// InOp
//===----------------------------------------------------------------------===//
LogicalResult parseInOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType operand;

  if (parser.parseOperand(operand))
    return failure();

  parser.resolveOperand(operand, result.operands);
  // alternatively parse Memref and create one of our types for it.

  Type riseType;
  if (parser.parseColonType(riseType))
    return failure();

  result.addTypes(riseType);
  return success();
}

//===----------------------------------------------------------------------===//
// OutOp
//===----------------------------------------------------------------------===//
LogicalResult parseOutOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType outputOperand;
  OpAsmParser::OperandType resultOperand;

  if (parser.parseOperand(outputOperand))
    return failure();

  parser.resolveOperand(outputOperand, result.operands);

  if (parser.parseLess() || parser.parseMinus())
    return failure();

  if (parser.parseOperand(resultOperand))
    return failure();

  parser.resolveOperand(resultOperand, result.operands);

  return success();
}

//===----------------------------------------------------------------------===//
// LambdaOp
//===----------------------------------------------------------------------===//
LogicalResult parseLambdaOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType arg;
  SmallVector<OpAsmParser::OperandType, 4> arguments;
  Type type;
  SmallVector<Type, 4> argumentTypes;
  FunType funType;
  Type outType;

  if (parser.parseLParen())
    return failure();

  int numArgs = 0;
  if (!succeeded(parser.parseOptionalRParen())) {
    do {
      if (parser.parseRegionArgument(arg) || parser.parseColonType(type))
        return failure();

      arguments.push_back(arg);
      argumentTypes.push_back(type);
      numArgs++;
    } while (succeeded(parser.parseOptionalComma()));
  }

  if (parser.parseRParen() || parser.parseArrow() || parser.parseType(outType))
    return failure();

  // build up type of this lambda:
  funType =
      FunType::get(builder.getContext(), argumentTypes[numArgs - 1], outType);
  for (int i = arguments.size() - 2; i >= 0; i--) {
    funType = FunType::get(builder.getContext(), argumentTypes[i], funType);
  }
  result.addTypes(funType);

  // Parse body of lambda
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, arguments, argumentTypes))
    return failure();

  LambdaOp::ensureTerminator(*body, builder, result.location);
  return success();
}

void LambdaOp::build(
    OpBuilder &builder, OperationState &result, FunType lambdaType,
    function_ref<Value(OpBuilder &, Location, MutableArrayRef<BlockArgument>)>
        bodyBuilder) {
  result.addTypes(lambdaType);

  Region *lambdaRegion = result.addRegion();
  Block *body = new Block();

  body->addArgument(lambdaType.getInput());
  while (lambdaType.getOutput().isa<FunType>()) {
    lambdaType = lambdaType.getOutput().dyn_cast<FunType>();
    body->addArgument(lambdaType.getInput());
  }
  lambdaRegion->push_back(body);

  if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(body);
    Value returnValue =
        bodyBuilder(builder, result.location, body->getArguments());
    builder.create<rise::ReturnOp>(returnValue.getLoc(),
                                   ValueRange{returnValue});
  } else {
    builder.create<rise::ReturnOp>(result.location, ValueRange{});
  }
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//
LogicalResult parseApplyOp(OpAsmParser &parser, OperationState &result) {
  // setting false here enables the requirement to explicitly give the type of
  // the applied function

  OpAsmParser::OperandType funOperand;
  FunType funType;
  SmallVector<OpAsmParser::OperandType, 4> arguments;
  SmallVector<Type, 4> argumentTypes;

  // parse function
  if (parser.parseOperand(funOperand))
    return failure();

  if (parser.resolveOperand(funOperand, result.operands))
    failure();
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
          << "expected a maximum of " << std::to_string(i)
          << " arguments for this function.";
      return failure();
    }
  }
  if (parser.resolveOperands(arguments, argumentTypes,
                             parser.getCurrentLocation(), result.operands))
    failure();

  result.addTypes(funType.getOutput());
  return success();
}

//===----------------------------------------------------------------------===//
// ParseLiteralOp
//===----------------------------------------------------------------------===//

LogicalResult parseLiteralOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  LiteralAttr attr;

  // type and value of literal
  if (parser.parseAttribute(attr, "literal", result.attributes))
    return failure();

  result.addTypes(attr.getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Map
//===----------------------------------------------------------------------===//

/// map: {n : nat} → {s t : data} → (s → t ) → n.s → n.t
LogicalResult parseMapSeqOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  NatAttr n;
  DataTypeAttr s, t;

  // optional lowering target
  if (failed(parseOptionalLoweringTargetAttribute(parser, result)))
    failure();

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
      FunType::get(builder.getContext(), s.getValue(), t.getValue()),
      FunType::get(
          builder.getContext(),
          ArrayType::get(builder.getContext(), n.getValue(), s.getValue()),
          ArrayType::get(builder.getContext(), n.getValue(), t.getValue()))));
  return success();
}

/// map: {n : nat} → {s t : data} → (s → t ) → n.s → n.t
LogicalResult parseMapParOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  NatAttr n;
  DataTypeAttr s, t;

  // optional lowering target
  if (failed(parseOptionalLoweringTargetAttribute(parser, result)))
    failure();

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
      FunType::get(builder.getContext(), s.getValue(), t.getValue()),
      FunType::get(
          builder.getContext(),
          ArrayType::get(builder.getContext(), n.getValue(), s.getValue()),
          ArrayType::get(builder.getContext(), n.getValue(), t.getValue()))));
  return success();
}

/// map: {n : nat} → {s t : data} → (s → t ) → n.s → n.t
LogicalResult parseMapOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  NatAttr n;
  DataTypeAttr s, t;

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
      FunType::get(builder.getContext(), s.getValue(), t.getValue()),
      FunType::get(
          builder.getContext(),
          ArrayType::get(builder.getContext(), n.getValue(), s.getValue()),
          ArrayType::get(builder.getContext(), n.getValue(), t.getValue()))));
  return success();
}

//===----------------------------------------------------------------------===//
// Reduce
//===----------------------------------------------------------------------===//

/// reduce: {n : nat} → {s t : data} → (s → t → t ) → t → n.s → t
LogicalResult parseReduceSeqOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  NatAttr n;
  DataTypeAttr s, t;

  // optional lowering target
  if (failed(parseOptionalLoweringTargetAttribute(parser, result)))
    failure();

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
          builder.getContext(), s.getValue(),
          FunType::get(builder.getContext(), t.getValue(), t.getValue())),
      FunType::get(builder.getContext(), t.getValue(),
                   FunType::get(builder.getContext(),
                                ArrayType::get(builder.getContext(),
                                               n.getValue(), s.getValue()),
                                t.getValue()))));

  return success();
}

//===----------------------------------------------------------------------===//
// Split
//===----------------------------------------------------------------------===//

/// split: (n:nat) → {m:nat} → {t:data} → nm.t → m.n.t
LogicalResult parseSplitOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  NatAttr n, m;
  DataTypeAttr t;

  if (parser.parseAttribute(n, "n", result.attributes))
    failure();

  if (parser.parseAttribute(m, "m", result.attributes))
    failure();

  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(
      FunType::get(builder.getContext(),
                   ArrayType::get(builder.getContext(),
                                  Nat::get(builder.getContext(),
                                           n.getValue().getIntValue() *
                                               m.getValue().getIntValue()),
                                  t.getValue()),
                   ArrayType::get(builder.getContext(), m.getValue(),
                                  ArrayType::get(builder.getContext(),
                                                 n.getValue(), t.getValue()))));

  return success();
}

//===----------------------------------------------------------------------===//
// Join
//===----------------------------------------------------------------------===//

/// join: {n m:nat} → {t:data} → n.m.t → nm.t
LogicalResult parseJoinOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  NatAttr n, m;
  DataTypeAttr t;

  if (parser.parseAttribute(n, "n", result.attributes))
    failure();

  if (parser.parseAttribute(m, "m", result.attributes))
    failure();

  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(
      FunType::get(builder.getContext(),
                   ArrayType::get(builder.getContext(), n.getValue(),
                                  ArrayType::get(builder.getContext(),
                                                 m.getValue(), t.getValue())),
                   ArrayType::get(builder.getContext(),
                                  Nat::get(builder.getContext(),
                                           n.getValue().getIntValue() *
                                               m.getValue().getIntValue()),
                                  t.getValue())));

  return success();
}

//===----------------------------------------------------------------------===//
// Transpose
//===----------------------------------------------------------------------===//

/// transpose: {n m:nat} → {t:data} → n.m.t → m.n.t
LogicalResult parseTransposeOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  NatAttr n, m;
  DataTypeAttr t;

  if (parser.parseAttribute(n, "n", result.attributes))
    failure();

  if (parser.parseAttribute(m, "m", result.attributes))
    failure();

  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(
      FunType::get(builder.getContext(),
                   ArrayType::get(builder.getContext(), n.getValue(),
                                  ArrayType::get(builder.getContext(),
                                                 m.getValue(), t.getValue())),
                   ArrayType::get(builder.getContext(), m.getValue(),
                                  ArrayType::get(builder.getContext(),
                                                 n.getValue(), t.getValue()))));

  return success();
}

//===----------------------------------------------------------------------===//
// Slide
//===----------------------------------------------------------------------===//

/// slide: {n:nat} → (sz sp:nat) → {t:data} → (sp*n+sz−sp).t → n.sz.t
LogicalResult parseSlideOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  NatAttr n, sz, sp;
  DataTypeAttr t;

  if (parser.parseAttribute(n, "n", result.attributes))
    failure();

  if (parser.parseAttribute(sz, "sz", result.attributes))
    failure();

  if (parser.parseAttribute(sp, "sp", result.attributes))
    failure();

  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(FunType::get(
      builder.getContext(),
      ArrayType::get(
          builder.getContext(),
          Nat::get(builder.getContext(),
                   sp.getValue().getIntValue() * n.getValue().getIntValue() +
                       sz.getValue().getIntValue() -
                       sp.getValue().getIntValue()),
          t.getValue()),
      ArrayType::get(
          builder.getContext(), n.getValue(),
          ArrayType::get(builder.getContext(), sz.getValue(), t.getValue()))));

  return success();
}

//===----------------------------------------------------------------------===//
// Pad
//===----------------------------------------------------------------------===//

/// padClamp: {n:nat} → (l r:nat) → {t:data} → n.t → (l+n+r).t
LogicalResult parsePadOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  NatAttr n, l, r;
  DataTypeAttr t;

  if (parser.parseAttribute(n, "n", result.attributes))
    failure();

  if (parser.parseAttribute(l, "l", result.attributes))
    failure();

  if (parser.parseAttribute(r, "r", result.attributes))
    failure();

  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(FunType::get(
      builder.getContext(),
      ArrayType::get(builder.getContext(), n.getValue(), t.getValue()),
      ArrayType::get(
          builder.getContext(),
          Nat::get(builder.getContext(), l.getValue().getIntValue() +
                                             n.getValue().getIntValue() +
                                             r.getValue().getIntValue()),
          t.getValue())));

  return success();
}

//===----------------------------------------------------------------------===//
// Tuple Ops
//===----------------------------------------------------------------------===//

/// zip: {n : nat} → {s t : data} → n.s → n.t → n.(s × t )
LogicalResult parseZipOp(OpAsmParser &parser, OperationState &result) {
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
      ArrayType::get(builder.getContext(), n.getValue(), s.getValue()),
      FunType::get(
          builder.getContext(),
          ArrayType::get(builder.getContext(), n.getValue(), t.getValue()),
          ArrayType::get(
              builder.getContext(), n.getValue(),
              Tuple::get(builder.getContext(), s.getValue(), t.getValue())))));
  return success();
}

/// tuple: {s t : data} → s → t → s × t
LogicalResult parseTupleOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  DataTypeAttr s, t;

  // type of first element
  if (parser.parseAttribute(s, "s", result.attributes))
    failure();

  // type of second element
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(
      FunType::get(builder.getContext(), s.getValue(),
                   FunType::get(builder.getContext(), t.getValue(),
                                Tuple::get(builder.getContext(), s.getValue(),
                                           t.getValue()))));
  return success();
}

/// fst: {s t : data} → s × t → s
LogicalResult parseFstOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  DataTypeAttr s, t;

  // type of first element
  if (parser.parseAttribute(s, "s", result.attributes))
    failure();

  // type of second element
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(
      FunType::get(builder.getContext(),
                   Tuple::get(builder.getContext(), s.getValue(), t.getValue()),
                   s.getValue()));
  return success();
}

/// snd: {s t : data} → s × t → t
LogicalResult parseSndOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  DataTypeAttr s, t;

  // type of first element
  if (parser.parseAttribute(s, "s", result.attributes))
    failure();

  // type of second element
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(
      FunType::get(builder.getContext(),
                   Tuple::get(builder.getContext(), s.getValue(), t.getValue()),
                   t.getValue()));
  return success();
}

//===----------------------------------------------------------------------===//
// Arithmetics
//===----------------------------------------------------------------------===//

/// add: {t : data} → t → t → t
LogicalResult parseAddOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  DataTypeAttr t;

  // type of summands
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(FunType::get(
      builder.getContext(), t.getValue(),
      FunType::get(builder.getContext(), t.getValue(), t.getValue())));
  return success();
}

/// mult: {t : data} → t → t → t
LogicalResult parseMulOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  DataTypeAttr t;

  // type of factors
  if (parser.parseAttribute(t, "t", result.attributes))
    failure();

  result.addTypes(FunType::get(
      builder.getContext(), t.getValue(),
      FunType::get(builder.getContext(), t.getValue(), t.getValue())));
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//
LogicalResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType value;
  Type type;

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

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static LogicalResult
parseOptionalLoweringTargetAttribute(OpAsmParser &parser,
                                     OperationState &result) {
  // parsing the optional attribute specifying a lowering target
  SmallVector<std::string, 4> loweringTargets = {"affine", "scf"};
  NamedAttrList attributesFromDict;

  if (succeeded(parser.parseOptionalAttrDict(attributesFromDict))) {
    if (!attributesFromDict.empty()) {
      bool validLowering = false;
      if (attributesFromDict.begin()->first.str() == "to") {
        if (StringAttr loweringAttr =
                attributesFromDict.begin()->second.dyn_cast<StringAttr>()) {
          for (std::string target : loweringTargets) {
            if (target == loweringAttr.getValue().str()) {
              validLowering = true;
              result.attributes.push_back(
                  attributesFromDict.getAttrs().front());
              return success();
            }
          }
        }
      }
      if (!validLowering) {
        std::string loweringTargetsString;
        for (std::string target : loweringTargets) {
          loweringTargetsString.append(target);
          loweringTargetsString.append(", ");
        }
        loweringTargetsString.pop_back();
        loweringTargetsString.pop_back();
        emitError(result.location) << "invalid lowering target specified. Use "
                                      "on of the following: to = "
                                   << loweringTargetsString;
        return failure();
      }
    }
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Rise/IR/Rise.cpp.inc"
} // end namespace rise
} // namespace mlir
