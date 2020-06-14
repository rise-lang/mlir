//===- Ops.td - Rise operation definitions ---------------*- tablegen -*-===//
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
// Defines RISE operations.
//
//===----------------------------------------------------------------------===//

#ifdef RISE_OPS
#else
#define RISE_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/Interfaces/SideEffectInterfaces.td"

def Rise_Dialect : Dialect {
  let name = "rise";
  let cppNamespace = "";
}

// Base class for Rise dialect ops.
class Rise_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<Rise_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

/// Type predicates
def NatType : Type<CPred<"$_self.isa<Nat>()">, "nat type">;
def TupleType : Type<CPred<"$_self.isa<Tuple>()">, "tuple type">;
def ArrayType : Type<CPred<"$_self.isa<ArrayType>()">, "array type">;
def ScalarType : Type<CPred<"$_self.isa<ScalarType>()">, "scalar type">;
def DataType
    : Type<
          Or<[TupleType.predicate, ArrayType.predicate, ScalarType.predicate]>>;
def NatOrDataType : Type<Or<[NatType.predicate, DataType.predicate]>>;

def FunType : Type<CPred<"$_self.isa<FunType>()">, "fun type">;
def DataTypeWrapper
    : Type<CPred<"$_self.isa<DataTypeWrapper>()">, "data fun type">;
def RiseType : Type<Or<[FunType.predicate, DataTypeWrapper.predicate]>>;

/// Attribute predicates
def DataTypeAttr
    : Attr<CPred<"$_self.isa<DataTypeAttr>()">, "datatype attribute"> {
  let storageType = [{DataTypeAttr}];
  let returnType = [{Type}];
}
def NatAttr : Attr<CPred<"$_self.isa<NatAttr>()">, "nat attribute"> {
  let storageType = [{NatAttr}];
  let returnType = [{Nat}];
}
def LiteralAttr : Attr<CPred<"$_self.isa<LiteralAttr>()">, "literal "
                                                           "attribute"> {
  let storageType = [{LiteralAttr}];
  let returnType = [{std::string}];
}

//===----------------------------------------------------------------------===//
// Rise Operations: Core Lambda Calculus
//===----------------------------------------------------------------------===//

def LambdaOp : Rise_Op<"lambda", [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "lambda";
  let description =
      "This operation is used for function abstraction in the RISE dialect. "
      "It associates a Region of one Block with a RISE FunType."
      "Hence, this operation gives a Region the meaning of a functional lambda "
      "expression.";
  let regions = (region SizedRegion<1> : $region);
  let results = (outs FunType : $lambdaType);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder &builder, OperationState &result, "
                            "FunType lambdaType, function_ref<void(OpBuilder "
                            "&, Location, MutableArrayRef<BlockArgument>)> "
                            "bodyBuilder = nullptr">];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def ApplyOp : Rise_Op<"apply"> {
  let summary = "apply";
  let description =
      "This operation is used for mirroring function application of lambda "
      "calculus. "
      "The fun can be a RISE lambda or any other value with a RISE function "
      "type, "
      "such as a RISE pattern or a partially applied function. "
      "The type of this operation consequently depends on the fun and the "
      "given arguments i.e full or partial application.";
  let arguments = (ins FunType : $fun, Variadic<AnyType> : $arguments);
  let results = (outs AnyType : $result);
  let parser = [{ return parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Rise Operations: Interoperability
//===----------------------------------------------------------------------===//

def InOp : Rise_Op<"in"> {
  let summary = "rise in";
  let description = "";
  let arguments = (ins AnyMemRef : $input);
  let results = (outs AnyType : $output);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def OutOp : Rise_Op<"out"> {
  let summary = "rise out";
  let arguments = (ins AnyType : $output, AnyType : $input);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def EmbedOp : Rise_Op<"embed", [IsolatedFromAbove]> {
  let summary = "rise embed";
  let arguments = (ins Variadic<ScalarType> : $exposedValues);  // This is not really used. We have region arguments, not "normal" arguments.
  let results = (outs AnyType : $wrapped);
  let regions = (region SizedRegion<1> : $region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
      "OpBuilder &builder, OperationState &result, "
      "Type wrapped, ValueRange exposedValues, function_ref<void(OpBuilder "
      "&, Location, MutableArrayRef<BlockArgument>)> "
      "bodyBuilder = nullptr">];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def ReturnOp : Rise_Op<"return", [Terminator]> {
  let summary = "rise terminator operation";
  let description =
      "This operation is used as a terminator for the region of a lambda and "
      "embed. "
      "We will experiment with using the standard return operation instead.";
  let arguments = (ins Variadic<AnyType> : $operands);
  let parser = [{ return parse$cppClass(parser, result); }];
  let builders = [OpBuilder<"OpBuilder &b, OperationState &result",
                            [{ build(b, result, llvm::None); }]>];
}

//===----------------------------------------------------------------------===//
// Rise Operations: Patterns
//===----------------------------------------------------------------------===//

def MapSeqOp : Rise_Op<"mapSeq"> {
  let summary = "mapSeq";
  let description =
      "This operation returns a function which expects a function from s to t "
      "and an array of length n. The function is to be applied sequentially to "
      "all elements of the array.";
  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t, OptionalAttr<StrAttr> : $to);
  let parser = [{ return parse$cppClass(parser, result); }];

  let results = (outs FunType : $mapSeqFun);
}

def MapParOp : Rise_Op<"mapPar"> {
  let summary = "mapPar";
  let description =
      "This operation returns a function which expects a function from s to t "
      "and an array of length n. The function is to be applied to all elements "
      "of the array in parallel.";
  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t, OptionalAttr<StrAttr> : $to);
  let parser = [{ return parse$cppClass(parser, result); }];

  let results = (outs FunType : $mapParFun);
}

def ReduceSeqOp : Rise_Op<"reduceSeq"> {
  let summary = "reduceSeq";
  let description =
      "This operation returns a function which expects an accumulator function "
      "from s to t, an accumulator of type t, and an array of size n. It "
      "returns "
      "a value of type t This can be used to reduce an array of size n with "
      "elements of type s to a value of type t";
  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t, OptionalAttr<StrAttr> : $to);
  let parser = [{ return parse$cppClass(parser, result); }];

  let results = (outs FunType : $reduceFun);
}

def ZipOp : Rise_Op<"zip"> {
  let summary = "rise zip operation";
  let description =
      "This operation returns a function which expects an array of size n with "
      "elements of type s and another array of size n and element type t. It "
      "returns an array of size n, containing tuples <s,t>. This is used to "
      "zip two arrays.";
  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t);
  let parser = [{ return parse$cppClass(parser, result); }];
  let results = (outs FunType : $zipFun);
}

def TupleOp : Rise_Op<"tuple"> {
  let summary = "rise tuple operation";
  let description =
      "This operation returns a function which expects two arguments of type "
      "s and t and returns a tuple of type <s,t>. It is used to create tuples.";
  let arguments = (ins DataTypeAttr : $s, DataTypeAttr : $t);
  let results = (outs FunType : $tupleFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def FstOp : Rise_Op<"fst"> {
  let summary = "rise fst operation";
  let description =
      "This operation returns a function which expects a tuple <s,t> and "
      "returns the first element of the tuple.";
  let arguments = (ins DataTypeAttr : $s, DataTypeAttr : $t);
  let results = (outs FunType : $fstFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def SndOp : Rise_Op<"snd"> {
  let summary = "rise snd operation";
  let description =
      "This operation returns a function which expects a tuple <s,t> and "
      "returns the second element of the tuple.";
  let arguments = (ins DataTypeAttr : $s, DataTypeAttr : $t);
  let results = (outs FunType : $sndFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def SplitOp : Rise_Op<"split"> {
  let summary = "split";
  let arguments = (ins NatAttr : $n, NatAttr : $m, DataTypeAttr : $t);
  let results = (outs FunType : $splitFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def JoinOp : Rise_Op<"join"> {
  let summary = "join";
  let arguments = (ins NatAttr : $n, NatAttr : $m, DataTypeAttr : $t);
  let results = (outs FunType : $joinFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def TransposeOp : Rise_Op<"transpose"> {
  let summary = "transpose";
  let arguments = (ins NatAttr : $n, NatAttr : $m, DataTypeAttr : $t);
  let results = (outs FunType : $tranposeFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def SlideOp : Rise_Op<"slide"> {
  let summary = "slide";
  let arguments = (ins NatAttr : $n, NatAttr : $sz, NatAttr : $sp, DataTypeAttr : $t);
  let results = (outs FunType : $slideFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def PadCstOp : Rise_Op<"pad"> {
  let summary = "pad";
  let arguments = (ins NatAttr : $n, NatAttr : $l, NatAttr : $q, DataTypeAttr : $t);
  let results = (outs FunType : $padFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def LiteralOp : Rise_Op<"literal"> {
  let summary = "literal";
  let arguments = (ins LiteralAttr : $literal);
  let results = (outs AnyType);
  let parser = [{ return parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Rise Operations: Intermediate Codegen
//===----------------------------------------------------------------------===//

def IdxOp : Rise_Op<"codegen.idx"> {
  let summary = "rise idx";
  let arguments = (ins DataType : $arg0, AnyType : $arg1);
  let results = (outs DataType : $output);
}

def CastOp : Rise_Op<"codegen.cast"> {
  let summary = "rise codegen.cast op";
  let arguments = (ins AnyType);
  let results = (outs DataType);
}

def BinaryOp : Rise_Op<"codegen.bin_op"> {
  let summary = "rise bin_op";
  let arguments = (ins StrAttr : $op, AnyType : $arg0, AnyType : $arg1);
  let results = (outs AnyType : $output);
  // let parser = [{ return parse$cppClass(parser, result); }];
}

def ZipIntermediateOp : Rise_Op<"codegen.zip"> {
  let summary = "rise zip_interm";
  let arguments = (ins AnyType : $lhs, AnyType : $rhs);
  let results = (outs AnyType : $output);
}
def FstIntermediateOp : Rise_Op<"codegen.fst"> {
  let summary = "rise fst_interm";
  let arguments = (ins AnyType : $value);
  let results = (outs AnyType : $output);
}

def SndIntermediateOp : Rise_Op<"codegen.snd"> {
  let summary = "rise snd_interm";
  let arguments = (ins AnyType : $value);
  let results = (outs AnyType : $output);
}

def TransposeIntermediateOp : Rise_Op<"codegen.transpose"> {
  let summary = "rise snd_interm";
  let arguments = (ins AnyType : $value);
  let results = (outs AnyType : $output);
}

def AssignOp : Rise_Op<"codegen.assign"> {
  let summary = "rise assign";
  let arguments = (ins AnyType : $value, AnyType : $assignee);
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

def AddOp : Rise_Op<"add"> {
  let summary = "rise add operation";
  let description =
      "This operation returns a function which expects two arguments of type "
      "t and returns their sum of type t. We will experiment with adopting "
      "the standard arithmetics operations.";
  let arguments = (ins DataTypeAttr : $t);
  let results = (outs FunType : $addFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def MulOp : Rise_Op<"mul"> {
  let summary = "rise mul operation";
  let description =
      "This operation returns a function which expects two arguments of type "
      "t and returns their product of type t. We will experiment with adopting "
      "the standard arithmetics operations.";
  let arguments = (ins DataTypeAttr : $t);
  let results = (outs FunType : $mulFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

#endif // RISE_OPS
