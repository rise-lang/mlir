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
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

using llvm::ArrayRef;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace mlir {
namespace rise {

#define DEBUG_TYPE "rise"

/// Dialect creation
RiseDialect::RiseDialect(mlir::MLIRContext *ctx) : mlir::Dialect("rise", ctx, TypeID::get<RiseDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Rise/IR/Rise.cpp.inc"
      >();
  ///      Types:                    Nats: Datatypes:
  addTypes<FunType, DataTypeWrapper, Nat, Tuple, ArrayType, ScalarType>();
  addAttributes<DataTypeAttr, NatAttr, LiteralAttr>();
}

void mlir::rise::RiseDialect::initialize() {
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

void dumpRiseExpression_value(Value val, SmallVector<BlockArgument, 4> &args) {
//  llvm::dbgs() << "valsize:" << args.size();
  for (int i = 0; i < args.size(); i++) {
    if (args[i] == val) {
      llvm::dbgs() << "x" << i;
      return;
    }
  }
  llvm::dbgs() << "x?";
//  emitWarning(val.getLoc()) << "Possible out of scope Value!";
}

void dumpRiseExpression_recurse(Operation *op, SmallVector<BlockArgument, 4> &args, bool omitApplyNodes, bool printBinderTypes, int indentLevel = 0) {
  auto indent = [&]() -> void {
    llvm::dbgs() << "\n";
    for (int i = 0; i < indentLevel * 4; i++) {
      llvm::dbgs() << " ";
    }
  };
  auto getDefiningOpOrNullptr = [&](Value val) -> Operation* {
    if (val.isa<OpResult>()) {
      return val.getDefiningOp();
    } else {
      return nullptr;
    }
  };
  auto getNestingLevelForLambda = [&](LambdaOp &lambda) -> int {
    int i = 0;
    LambdaOp parentLambda = lambda;
    while (parentLambda = parentLambda.getParentOfType<LambdaOp>()) {
      i++;
    }
    return i;
  };
  auto getNestingLevelForEmbed = [&](EmbedOp &embed) -> int {
    int i = 0;
    EmbedOp parentEmbed = embed;
    while (parentEmbed = parentEmbed.getParentOfType<EmbedOp>()) {
      i++;
    }
    return i;
  };
  auto dumpOperands = [&](Operation *op) -> void {
    int firstOpToDump = 0;
    if (omitApplyNodes && isa<ApplyOp>(op)) {
      dumpRiseExpression_recurse(op->getOperand(0).getDefiningOp(), args, omitApplyNodes, printBinderTypes, indentLevel);
      llvm::dbgs() << "(";
      firstOpToDump++;
    }
    for (int i = firstOpToDump; i < op->getNumOperands(); i++) {
      if (op->getOperand(i).isa<OpResult>()) {
        dumpRiseExpression_recurse(op->getOperand(i).getDefiningOp(), args, omitApplyNodes, printBinderTypes, indentLevel);
      } else {
        dumpRiseExpression_value(op->getOperand(i), args);
      }
      if (i < op->getNumOperands()-1) llvm::dbgs() << ",";
    }
  };
  auto changeColor = [&](Operation *op) -> void {
    Operation *opColored = op;
    // for rise patterns the color is only in its apply operation
    if (!op->getUsers().empty() && !isa<EmbedOp>(op) && !isa<ApplyOp>(op) && op->getDialect()->getNamespace() == RiseDialect::getDialectNamespace()) {
      opColored = *op->getUsers().begin();
    }

    if (auto embeddingAttr = opColored->getAttrOfType<ArrayAttr>("ksc.color")) {
      // interpreting first three components of color as rgb
      int r = embeddingAttr.getValue()[0].dyn_cast<FloatAttr>().getValue().convertToFloat() * 255;
      int g = embeddingAttr.getValue()[1].dyn_cast<FloatAttr>().getValue().convertToFloat() * 255;
      int b = embeddingAttr.getValue()[1].dyn_cast<FloatAttr>().getValue().convertToFloat() * 255;
      llvm::dbgs() << "\x1b[38;2;" << r << ";" << g << ";" << b << "m";
    }
  };
  auto resetColor = [&]() -> void {
    llvm::dbgs() << "\x1b[0m";
  };
  if (op == nullptr) return;
  if (auto out = dyn_cast<OutOp>(op)) {
    llvm::dbgs() << "out = (\n";
    dumpRiseExpression_recurse(getDefiningOpOrNullptr(out.getOperand(1)), args, omitApplyNodes, printBinderTypes, indentLevel);
    llvm::dbgs() << ")\n";

    return;
  }
  if (auto apply = dyn_cast<ApplyOp>(op)) {
    changeColor(op);
    if (!omitApplyNodes) llvm::dbgs() << "App(";
    resetColor();
    dumpOperands(apply);
    llvm::dbgs() << ")";
    return;
  }
  if (auto lambda = dyn_cast<LambdaOp>(op)) {
    // print args of this lambda
    llvm::dbgs() << "λ(";
    for (auto arg : lambda.region().front().getArguments()) {
      args.push_back(arg);
      llvm::dbgs() << "x" << args.size()-1;
      if (printBinderTypes) {
        llvm::dbgs() << " : ";
        arg.getType().print(llvm::dbgs());
      }
      if (arg != lambda.region().front().getArguments().back()) llvm::dbgs() << ",";
    }
    llvm::dbgs() << " =>";
    indentLevel++;
    indent();
//    for (int i = 0; i < nestingLevel; i++) {
//      llvm::dbgs() << "  ";
//    }
    dumpRiseExpression_recurse(getDefiningOpOrNullptr(lambda.region().front().getTerminator()->getOperand(0)), args, omitApplyNodes, printBinderTypes, indentLevel);
    llvm::dbgs() << ")";
    indentLevel--;
    indent();
    return;
  }
  if (auto embed = dyn_cast<EmbedOp>(op)) {
    changeColor(embed);
    llvm::dbgs() << "embed(";
    resetColor();
    for (auto arg : embed.region().front().getArguments()) {
      args.push_back(arg);
      llvm::dbgs() << "e" << args.size()-1;
      if (printBinderTypes) {
        llvm::dbgs() << " : ";
        arg.getType().print(llvm::dbgs());
      }
      if (arg != embed.region().front().getArguments().back()) llvm::dbgs() << ",";
    }
    llvm::dbgs() << " => {";
    indentLevel++;
    indent();
    dumpRiseExpression_recurse(getDefiningOpOrNullptr(embed.region().front().getTerminator()->getOperand(0)), args, omitApplyNodes, printBinderTypes, indentLevel);
    indent();
    llvm::dbgs() << "}, ";
    dumpOperands(embed);
    llvm::dbgs() << ")";
    indentLevel--;
    return;
  }
  if (auto split = dyn_cast<SplitOp>(op)) {
    changeColor(split);
    llvm::dbgs() << "split" << split.n().getIntValue() << ")";
    resetColor();
    return;
  }
  if (auto literal = dyn_cast<LiteralOp>(op)) {
    changeColor(literal);
    llvm::dbgs() << "l(" << literal.literal() << ")";
    resetColor();
    return;
  }
  // not first class printable op from RISE
  if (isa<RiseDialect>(op->getDialect())) {
    changeColor(op);
    llvm::dbgs() << op->getName().getStringRef().drop_front(5) << "";
    resetColor();
//    dumpOperands(op);
    return;
  }
  if (FuncOp funcOp = dyn_cast<FuncOp>(op)) {
    funcOp.walk([&](Operation* op){
      if (isa<OutOp>(op)) {
        RiseDialect::dumpRiseExpression(op);
        llvm::dbgs() << "\n";
      }
    });
    return;
  }
  if (ModuleOp moduleOp = dyn_cast<ModuleOp>(op)) {
    moduleOp.walk([&](Operation* op){
      if (isa<OutOp>(op)) {
        RiseDialect::dumpRiseExpression(op);
        llvm::dbgs() << "\n";
      }
    });
    return;
  }
  // not first class printable op from another Dialect
  changeColor(op);
  llvm::dbgs() << op->getName();
  resetColor();
  if (op->getNumOperands() > 0) {
    llvm::dbgs() << "(";
    dumpOperands(op);
    llvm::dbgs() << ")";
  }
}

void RiseDialect::dumpRiseExpression(Operation *op, bool omitApplyNodes, bool printBinderTypes) {
  SmallVector<BlockArgument, 4> args;
  dumpRiseExpression_recurse(op, args, omitApplyNodes, printBinderTypes);
}

int RiseDialect::getCostForSubexpression(Operation *op, bool propagateUp) {
  if (!op) return -1;
  std::function<int(Operation*, int)> setCost = [&](Operation *op, int cost) -> int {
    if (propagateUp) {
      Operation *parentOp = op;
      int oldOpCost = 0;
      if (op->getAttrOfType<IntegerAttr>("rlo.cost"))
        oldOpCost = op->getAttrOfType<IntegerAttr>("rlo.cost").getInt();
      while(parentOp = parentOp->getParentOp()) {
        if (!parentOp->getAttrOfType<IntegerAttr>("rlo.cost"))
          break;
        int oldParentCost = parentOp->getAttrOfType<IntegerAttr>("rlo.cost").getInt();
        setCost(parentOp, oldParentCost - oldOpCost + cost);
      }
    }
    op->setAttr("rise.cost", IntegerAttr::get(IntegerType::get(op->getContext(), 32), cost));
    return cost;
  };

  // cost of rise operations with a region i.e lambda, embed, loweringUnit
  if (op->getDialect()->getNamespace() == RiseDialect::getDialectNamespace() &&
      op->getNumRegions() == 1) {
    int riseRegionCost = 0;
    op->getRegion(0).walk_shallow([&](Operation *nestedOp) {
      LLVM_DEBUG({llvm::dbgs() << "Getting nested cost of op: " << nestedOp->getName() << " for region op:" << op->getName() << "\n";});
      riseRegionCost += RiseDialect::getCostForSubexpression(nestedOp);
    });
    return setCost(op, riseRegionCost);
  }
  if (ApplyOp apply = dyn_cast<ApplyOp>(op)) {
    if (MapSeqOp mapSeqOp = dyn_cast<MapSeqOp>(apply.fun().getDefiningOp())) {
      int lambdaCost = RiseDialect::getCostForSubexpression(
          apply->getOperand(1).getDefiningOp<LambdaOp>());
      return setCost(op, 1 + mapSeqOp.n().getIntValue() * lambdaCost);
    } else if (ReduceSeqOp reduceSeqOp =
                   dyn_cast<ReduceSeqOp>(apply.fun().getDefiningOp())) {
      int lambdaCost = RiseDialect::getCostForSubexpression(
          apply->getOperand(1).getDefiningOp<LambdaOp>());
      return setCost(op, 2 + reduceSeqOp.n().getIntValue() * lambdaCost);
    } else if (LambdaOp lambda = dyn_cast<LambdaOp>(apply.fun().getDefiningOp())) {
      return setCost(op, RiseDialect::getCostForSubexpression(lambda));
    } else if (JoinOp join = dyn_cast<JoinOp>(apply.fun().getDefiningOp())) {
      return setCost(op, 4);
    } else if (SplitOp split = dyn_cast<SplitOp>(apply.fun().getDefiningOp())) {
      return setCost(op, 4);
    } else if (SlideOp slide = dyn_cast<SlideOp>(apply.fun().getDefiningOp())) {
      return setCost(op, 2);
    } else if (PadOp pad = dyn_cast<PadOp>(apply.fun().getDefiningOp())) {
      return setCost(op, 3);
    }
  }
  // only applied rise operations have a cost
  if (op->getDialect()->getNamespace() == RiseDialect::getDialectNamespace())
    return setCost(op, 0);

  return setCost(op, 1);
}

DataType RiseDialect::getAsDataType(Type type) {
  if (auto dataType = type.dyn_cast<DataType>()) return dataType;
  if (auto scalarType = type.dyn_cast<ScalarType>()) return scalarType;
  if (auto arrayType = type.dyn_cast<ArrayType>()) return arrayType;
  if (auto tupleType = type.dyn_cast<Tuple>()) return tupleType;
  llvm::llvm_unreachable_internal("type is not a subclass of Rise Datatype");
  return nullptr;
}

DataType RiseDialect::getFunTypeOutput(FunType funType) {
  auto output = funType.getOutput();
  while (auto type = output.dyn_cast<FunType>()) {
    output = output.cast<FunType>().getOutput();
  }
  return RiseDialect::getAsDataType(output);
}

} // end namespace rise
} // end namespace mlir