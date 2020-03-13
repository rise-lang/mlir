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

def IntType : Type<CPred<"$_self.isa<Int>()">, "int type">;
def FloatType : Type<CPred<"$_self.isa<Float>()">, "float type">;
def TupleType : Type<CPred<"$_self.isa<Tuple>()">, "tuple type">;
def ArrayType : Type<CPred<"$_self.isa<ArrayType>()">, "array type">;
def DataType : Type<Or<[
  IntType.predicate, FloatType.predicate, TupleType.predicate,
  ArrayType.predicate
]>>;
def NatOrDataType : Type<Or<[NatType.predicate, DataType.predicate]>>;

def FunType : Type<CPred<"$_self.isa<FunType>()">, "fun type">;
def DataTypeWrapper
    : Type<CPred<"$_self.isa<DataTypeWrapper>()">, "data fun type">;
def RiseType : Type<Or<[FunType.predicate, DataTypeWrapper.predicate]>>;

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
// Rise Operations
//===----------------------------------------------------------------------===//

def RiseContinuationTranslation : Rise_Op<"cont"> {
  let summary = "rise cont";
  let description =
      "A marker where to continue with the lowering to imperative";
  let arguments = (ins AnyType : $cont);
  let results = (outs AnyType);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def RiseFunOp : Rise_Op<"fun", [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "rise fun";
  let regions = (region SizedRegion<1> : $region);
  //  let arguments = (ins AnyType:$out, Variadic<AnyType>:$inputs);
  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$inputs);  // This is not really used. We have region arguments, not "normal" arguments.
  //  let results = (outs AnyType : $output);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def RiseOutOp : Rise_Op<"out"> {
  let summary = "rise out";
  let arguments = (ins AnyType : $input);
  let results = (outs DataType : $output);
  let parser = [{ return parse$cppClass(parser, result); }];
}

// def RiseInOp : Rise_Op<"in", [SingleBlockImplicitTerminator<"InOp">]> {
//  let summary = "rise in";
//  let arguments = (ins MemrefType:$input);
//  let results = (outs DataType:$output);
//  let parser = [{ return parse$cppClass(parser, result); }];
//}

def LiteralOp : Rise_Op<"literal", [NoSideEffect]> {
  let summary = "literal";
  let arguments = (ins LiteralAttr : $literal);
  let results = (outs RiseType);
  //    let extraClassDeclaration = [{LiteralAttr getValue() { return
  //    getAttr("value"); }
  //        /// Returns true if a constant operation can be built with the given
  //        value
  //        /// and result type.
  //        static bool isBuildableWith(LiteralAttr value, mlir::Type type);
  //    }];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def LambdaOp : Rise_Op<"lambda", [SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "lambda";
  let description =
      "This operation is used for function abstraction in the RISE dialect. "
      "It associates a Region of one Block with a RISE FunType."
      "Hence, this operation gives a Region the meaning of a functional lambda "
      "expression.";
  let regions = (region SizedRegion<1> : $region);
  let results = (outs FunType : $lambda_result);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def ApplyOp : Rise_Op<"apply", [NoSideEffect]> {
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

def MapOp : Rise_Op<"map", [NoSideEffect]> {
  let summary = "map";
  let description =
      "This operation returns a function which expects a function from s to t "
      "and an array of length n. The function is to be applied to all elements "
      "of the array.";
  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t);
  let parser = [{ return parse$cppClass(parser, result); }];

  let results = (outs FunType : $mapFun);
}

def ReduceOp : Rise_Op<"reduce", [NoSideEffect]> {
  let summary = "reduce";
  let description =
      "This operation returns a function which expects an accumulator function "
      "from s to t, an accumulator of type t, and an array of size n. It "
      "returns "
      "a value of type t This can be used to reduce an array of size n with "
      "elements of type s to a value of type t";
  let arguments = (ins NatAttr : $n, DataTypeAttr : $s, DataTypeAttr : $t);
  let parser = [{ return parse$cppClass(parser, result); }];

  let results = (outs FunType : $reduceFun);
}

def ZipOp : Rise_Op<"zip", [NoSideEffect]> {
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

def TupleOp : Rise_Op<"tuple", [NoSideEffect]> {
  let summary = "rise tuple operation";
  let description =
      "This operation returns a function which expects two arguments of type "
      "s and t and returns a tuple of type <s,t>. It is used to create tuples.";
  let arguments = (ins DataTypeAttr : $s, DataTypeAttr : $t);
  let results = (outs FunType : $tupleFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def FstOp : Rise_Op<"fst", [NoSideEffect]> {
  let summary = "rise fst operation";
  let description =
      "This operation returns a function which expects a tuple <s,t> and "
      "returns the first element of the tuple.";
  let arguments = (ins DataTypeAttr : $s, DataTypeAttr : $t);
  let results = (outs FunType : $fstFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def SndOp : Rise_Op<"snd", [NoSideEffect]> {
  let summary = "rise snd operation";
  let description =
      "This operation returns a function which expects a tuple <s,t> and "
      "returns the second element of the tuple.";
  let arguments = (ins DataTypeAttr : $s, DataTypeAttr : $t);
  let results = (outs FunType : $sndFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def ReturnOp : Rise_Op<"return", [Terminator]> {
  let summary = "rise terminator operation";
  let description =
      "This operation is used as a terminator for the region of a lambda. "
      "We will experiment with using the standard return operation instead.";
  let arguments = (ins Variadic<AnyType> : $operands);
  let parser = [{ return parse$cppClass(parser, result); }];
  let builders = [OpBuilder<"Builder *b, OperationState &result",
                            [{ build(b, result, llvm::None); }]>];
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

def AddOp : Rise_Op<"add", [NoSideEffect]> {
  let summary = "rise add operation";
  let description =
      "This operation returns a function which expects two arguments of type "
      "t and returns their sum of type t. We will experiment with adopting "
      "the standard arithmetics operations.";
  let arguments = (ins DataTypeAttr : $t);
  let results = (outs FunType : $addFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

def MultOp : Rise_Op<"mult", [NoSideEffect]> {
  let summary = "rise mult operation";
  let description =
      "This operation returns a function which expects two arguments of type "
      "t and returns their product of type t. We will experiment with adopting "
      "the standard arithmetics operations.";
  let arguments = (ins DataTypeAttr : $t);
  let results = (outs FunType : $multFun);
  let parser = [{ return parse$cppClass(parser, result); }];
}

#endif // RISE_OPS
