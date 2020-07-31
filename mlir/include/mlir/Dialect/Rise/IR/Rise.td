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
def FunType : Type<CPred<"$_self.isa<FunType>()">, "fun type">;

/// Attribute predicates
def DataTypeAttr
    : Attr<CPred<"$_self.isa<DataTypeAttr>()">, "datatype attribute"> {
  let storageType = [{DataTypeAttr}];
  let returnType = [{DataType}];
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
                            "FunType lambdaType, function_ref<Value(OpBuilder "
                            "&, Location, MutableArrayRef<BlockArgument>)> "
                            "bodyBuilder = nullptr">];
  let parser = [{ return parse$cppClass(parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
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

def LoweringUnitOp
    : Rise_Op<"lowering_unit", [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "rise lowering unit";
  let description = "Represents a unit of RISE operations to be lowered "
                    "independently of other RISE operations.";
  let regions = (region SizedRegion<1> : $region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder &builder, OperationState &result, "
                            "function_ref<void(OpBuilder &, Location)> "
                            "bodyBuilder">];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def InOp : Rise_Op<"in"> {
  let summary = "rise in";
  let description =
      "Expects a Value of Type MemRef and associates it with a Rise DataType.";
  let arguments = (ins AnyMemRef : $input);
  let results = (outs DataType : $output);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def OutOp : Rise_Op<"out"> {
  let summary = "rise out";
  let description =
      "Specifies a Memref to write a rise Value to. This "
      "operation is used as entry point for lowering a rise program. Currently "
      "this operation has to appear exactly once in a Rise translation unit.";
  let arguments = (ins AnyMemRef : $output, AnyType : $input);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def EmbedOp : Rise_Op<"embed", [IsolatedFromAbove]> {
  let summary = "rise embed";
  let description =
      "This operation is used for explicitly embedding "
      "operations of other dialects inside a rise translation "
      "unit while bridging between the type systems. The arguments of type "
      "!rise.scalar<wrappedType> given to this operation are exposed inside "
      "its region with their corresponding wrappedTypes.";
  let arguments = (ins Variadic<ScalarType> : $exposedValues);
  let results = (outs AnyType : $wrapped);
  let regions = (region SizedRegion<1> : $region);
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
      "OpBuilder &builder, OperationState &result, "
      "Type wrapped, ValueRange exposedValues, function_ref<Value(OpBuilder "
      "&, Location, MutableArrayRef<BlockArgument>)> "
      "bodyBuilder = nullptr">];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def ReturnOp : Rise_Op<"return", [Terminator]> {
  let summary = "rise terminator operation";
  let description = "This operation is used as a terminator for the region of "
                    "a rise.lambda and "
                    "rise.embed.";
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
      "The mapSeq operation represents the functional mapSeq pattern."
      "Its application expects a function from s to t and an array of length n "
      "with elements of type s."
      "The function is to be applied sequentially to all elements of the "
      "array.";

  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t, OptionalAttr<StrAttr> : $to);
  let parser = [{ return parse$cppClass(parser, result); }];

  let results = (outs FunType : $mapSeqFun);
}

def MapParOp : Rise_Op<"mapPar"> {
  let summary = "mapPar";
  let description =
      "The mapPar operation represents the functional mapPar pattern."
      "Its application expects a function from s to t and an array of length n "
      "with elements of type s."
      "The function is to be applied to all elements of the array in parallel "
      "(note: This is currently not lowered to parallel code).";
  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t, OptionalAttr<StrAttr> : $to);
  let parser = [{ return parse$cppClass(parser, result); }];

  let results = (outs FunType : $mapParFun);
}

def MapOp : Rise_Op<"map"> {
  let summary = "map";
  let description =
      "The mapSeq operation represents the functional map pattern."
      "Its application expects a function from s to t and an array of length n "
      "with elements of type s."
      "The function is to be applied sequentially to all elements of the "
      "array. This operation will never be lowered to a loop, but rather "
      "affect how values are read from the array.";
  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t);
  let parser = [{ return parse$cppClass(parser, result); }];

  let results = (outs FunType : $mapFun);
}

def ReduceSeqOp : Rise_Op<"reduceSeq"> {
  let summary = "reduceSeq";
  let description =
      "The reduceSeq operation represents the functional reduceSeq pattern. "
      "Its application expects an accumulator function from s to t, an "
      "accumulator of type t, and an array of size n. It returns a value of "
      "type t. "
      "This can be used to reduce an array of size n with elements of type s "
      "to a value of type t";
  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t, OptionalAttr<StrAttr> : $to);
  let parser = [{ return parse$cppClass(parser, result); }];

  let results = (outs FunType : $reduceFun);
}

def ZipOp : Rise_Op<"zip"> {
  let summary = "rise zip operation";
  let description =
      "The zip operation represents the functional zip pattern. Its "
      "application expects an array of size n with elements of type s and an "
      "array of size n and element type t. It returns an array of size n, "
      "containing tuples <s,t>."
      "This is used to zip two arrays.";
  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t);
  let parser = [{ return parse$cppClass(parser, result); }];
  let results = (outs FunType : $zipFun);
}

def TupleOp : Rise_Op<"tuple"> {
  let summary = "rise tuple operation";
  let description =
      "The tuple operation represents the functional tuple pattern. Its "
      "application expects two arguments of type "
      "s and t and returns a tuple of type <s,t>. It is used to create tuples.";
  let arguments = (ins DataTypeAttr : $s, DataTypeAttr : $t);
  let results = (outs FunType : $tupleFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def FstOp : Rise_Op<"fst"> {
  let summary = "rise fst operation";
  let description = "The fst operation represents the functional fst pattern. "
                    "Its application expects a tuple <s,t> and "
                    "returns the first element of the tuple.";
  let arguments = (ins DataTypeAttr : $s, DataTypeAttr : $t);
  let results = (outs FunType : $fstFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def SndOp : Rise_Op<"snd"> {
  let summary = "rise snd operation";
  let description = "The snd operation represents the functional snd pattern. "
                    "Its application expects a tuple <s,t> and "
                    "returns the second element of the tuple.";
  let arguments = (ins DataTypeAttr : $s, DataTypeAttr : $t);
  let results = (outs FunType : $sndFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def SplitOp : Rise_Op<"split"> {
  let summary = "split";
  let description =
      "The split operation represents the functional split pattern. Its "
      "application expects an array of size (n * m) and returns a nested array "
      "of "
      "length (m / n) where every element is an array of size n. The value of "
      "the "
      "elements of the input array remains the same.";
  let arguments = (ins NatAttr : $n, NatAttr : $m, DataTypeAttr : $t);
  let results = (outs FunType : $splitFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def JoinOp : Rise_Op<"join"> {
  let summary = "join";
  let description =
      "The join operation represents the functional join pattern. Its "
      "application expects a 2-dimensional array of size (n x m) and returns a "
      "flattened 1-dimensional array of size (n * m).";
  let arguments = (ins NatAttr : $n, NatAttr : $m, DataTypeAttr : $t);
  let results = (outs FunType : $joinFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def TransposeOp : Rise_Op<"transpose"> {
  let summary = "transpose";
  let description =
      "The transpose operation represents the functional transpose pattern. "
      "Its application expects a 2-dimensional array of size (n x m) and "
      "returns a "
      "2-dimensional array of size (m x n)";
  let arguments = (ins NatAttr : $n, NatAttr : $m, DataTypeAttr : $t);
  let results = (outs FunType : $tranposeFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def SlideOp : Rise_Op<"slide"> {
  let summary = "slide";
  let description =
      "The slide operation represents the functional slide pattern. Its "
      "application expects an array of size (sp*n+sz−sp) and returns a "
      "2-dimensional array of size (n x sz). In essence, this operation "
      "extracts neighbourhoods of size sz with stride sp from the given array.";
  let arguments = (ins NatAttr
                   : $n, NatAttr
                   : $sz, NatAttr
                   : $sp, DataTypeAttr
                   : $t);
  let results = (outs FunType : $slideFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def PadOp : Rise_Op<"pad"> {
  let summary = "pad";
  let description =
      "The pad operation represents the functional padClamp pattern. Its "
      "application expects an array of size n and returns an array of size (l "
      "+ n + r) by duplicating the boundary values.";
  let arguments = (ins NatAttr
                   : $n, NatAttr
                   : $l, NatAttr
                   : $r, DataTypeAttr
                   : $t);
  let results = (outs FunType : $padFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def LiteralOp : Rise_Op<"literal"> {
  let summary = "literal";
  let description = "The literal operation creates a literal of type and value "
                    "specified via a LiteralAttr.";
  let arguments = (ins LiteralAttr : $literal);
  let results = (outs AnyType);
  let parser = [{ return parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// Rise Operations: Intermediate Codegen
//===----------------------------------------------------------------------===//

def IdxOp : Rise_Op<"codegen.idx"> {
  let summary = "rise codegen.idx";
  let description =
      "The codegen.idx operation represents the indexing into a given "
      "array using an indexing value iv.";
  let arguments = (ins DataType : $array, Index : $iv);
  let results = (outs DataType : $output);
}

def CastOp : Rise_Op<"codegen.cast"> {
  let summary = "rise codegen.cast op";
  let description =
      "The codegen.cast operation associates a Value with a Rise DataType.";
  let arguments = (ins AnyType);
  let results = (outs DataType);
}

def MapReadIntermediateOp : Rise_Op<"codegen.mapread"> {
  let summary = "rise codegen.mapread";
  let description =
      "The codegen.mapread operation enables a 2-dimensional "
      "read from a 1-dimensional array. The actual indexing "
      "specified by the function f. The placeholder is replaced with the "
      "required codegen.idx in the second phase of the lowering process.";
  let arguments = (ins NatAttr
                   : $n, DataTypeAttr
                   : $s, DataTypeAttr
                   : $t, AnyType
                   : $f, AnyType
                   : $placeholder, AnyType
                   : $array);
  let results = (outs AnyType : $output);
}

def PlaceholderOp : Rise_Op<"codegen.placeholder"> {
  let summary = "rise codegen.placeholder";
  let description =
      "The codegen.placeholder operation is a placeholder for a codegen.idx "
      "operation, which will be introduced in the second phase of the lowering "
      "process while lowering codegen.mapread.";
  let results = (outs AnyType : $output);
}

def ZipIntermediateOp : Rise_Op<"codegen.zip"> {
  let summary = "rise zip_interm";
  let description = "The codegen.zip operation is the codegen equivalent of "
                    "the zip operation.";
  let arguments = (ins AnyType : $lhs, AnyType : $rhs);
  let results = (outs AnyType : $output);
}

def FstIntermediateOp : Rise_Op<"codegen.fst"> {
  let summary = "rise fst_interm";
  let description = "The codegen.fst operation is the codegen equivalent of "
                    "the fst operation.";
  let arguments = (ins AnyType : $value);
  let results = (outs AnyType : $output);
}

def SndIntermediateOp : Rise_Op<"codegen.snd"> {
  let summary = "rise snd_interm";
  let description = "The codegen.snd operation is the codegen equivalent of "
                    "the snd operation.";
  let arguments = (ins AnyType : $value);
  let results = (outs AnyType : $output);
}

def SplitIntermediateOp : Rise_Op<"codegen.split"> {
  let summary = "rise split_interm";
  let description = "The codegen.split operation is the codegen equivalent of "
                    "the split operation used in a reading access.";
  let arguments = (ins AnyType
                   : $value, NatAttr
                   : $n, NatAttr
                   : $m, DataTypeAttr
                   : $t);
  let results = (outs AnyType : $output);
}

def SplitAccIntermediateOp : Rise_Op<"codegen.splitAcc"> {
  let summary = "rise splitAcc_interm";
  let description =
      "The codegen.splitAcc operation is the codegen equivalent of "
      "the split operation used in a writing access.";
  let arguments = (ins AnyType
                   : $value, NatAttr
                   : $n, NatAttr
                   : $m, DataTypeAttr
                   : $t);
  let results = (outs AnyType : $output);
}

def JoinIntermediateOp : Rise_Op<"codegen.join"> {
  let summary = "rise join_interm";
  let description = "The codegen.join operation is the codegen equivalent of "
                    "the join operation used in a reading access.";
  let arguments = (ins AnyType
                   : $value, NatAttr
                   : $n, NatAttr
                   : $m, DataTypeAttr
                   : $t);
  let results = (outs AnyType : $output);
}

def JoinAccIntermediateOp : Rise_Op<"codegen.joinAcc"> {
  let summary = "rise joinAcc_interm";
  let description = "The codegen.join operation is the codegen equivalent of "
                    "the join operation used in a writing access.";
  let arguments = (ins AnyType
                   : $value, NatAttr
                   : $n, NatAttr
                   : $m, DataTypeAttr
                   : $t);
  let results = (outs AnyType : $output);
}

def TransposeIntermediateOp : Rise_Op<"codegen.transpose"> {
  let summary = "rise transpose_interm";
  let description =
      "The codegen.transpose operation is the codegen equivalent of "
      "the transpose operation.";
  let arguments = (ins AnyType
                   : $value, NatAttr
                   : $n, NatAttr
                   : $m, DataTypeAttr
                   : $t);
  let results = (outs AnyType : $output);
}

def TransposeAccIntermediateOp : Rise_Op<"codegen.transposeStore"> {
  let summary = "rise transposeAcc_interm";
  let description =
      "The codegen.transpose operation is the codegen equivalent of "
      "the transpose operation used in a writing access.";
  let arguments = (ins AnyType
                   : $value, NatAttr
                   : $n, NatAttr
                   : $m, DataTypeAttr
                   : $t);
  let results = (outs AnyType : $output);
}

def SlideIntermediateOp : Rise_Op<"codegen.slide"> {
  let summary = "rise slide_interm";
  let description = "The codegen.slide operation is the codegen equivalent of "
                    "the slide operation.";
  let arguments = (ins AnyType
                   : $value, NatAttr
                   : $n, NatAttr
                   : $sz, NatAttr
                   : $sp, DataTypeAttr
                   : $t);
  let results = (outs AnyType : $output);
}

def PadIntermediateOp : Rise_Op<"codegen.pad"> {
  let summary = "rise pad_interm";
  let description = "The codegen.pad operation is the codegen equivalent of "
                    "the pad operation.";
  let arguments = (ins AnyType
                   : $array, NatAttr
                   : $n, NatAttr
                   : $l, NatAttr
                   : $r, DataTypeAttr
                   : $t);
  let results = (outs AnyType : $output);
}

def AssignOp : Rise_Op<"codegen.assign"> {
  let summary = "rise assign";
  let description = "The codegen.assign operation models an assignment of "
                    "value to assignee.";
  let arguments = (ins AnyType : $value, AnyType : $assignee);
}

#endif // RISE_OPS
