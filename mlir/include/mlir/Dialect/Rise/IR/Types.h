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

#ifndef MLIR_RISE_TYPES_H
#define MLIR_RISE_TYPES_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Rise Dialect //////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace mlir {
namespace rise {

namespace detail {
struct ArrayTypeStorage;
struct RiseFunTypeStorage;
struct RiseDataTypeWrapperStorage;
struct RiseTupleTypeStorage;
struct RiseNatStorage;
} // namespace detail

/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum RiseTypeKind {
  // The enum starts at the range reserved for this dialect.
  RISE_TYPE = mlir::Type::FIRST_RISE_TYPE,
  RISE_FUNTYPE,
  RISE_DATATYPE_WRAPPER,
  RISE_NAT,
  RISE_DATATYPE,
  RISE_INT,
  RISE_FLOAT,
  RISE_TUPLE,
  RISE_ARRAY,
};

/// RISE type structure:
///                      +----------+
///                      |mlir::Type|
///                      +-+---+--+-+
///                        ^   ^  ^
///              +---------+   |  +----------+
///              |             |             |
///          +---+----+      +-+-+      +----+---+
///          |DataType|      |Nat|      |RiseType|
///          ++------++      +---+      +--+--+--+
///           ^      ^                     ^  ^
///           |      |                     |  |
///   +-------+      +-------+          +--+  +-------+
///   |       |      |       |          |             |
/// +-+-+ +---+-+ +--+--+ +--+--+   +---+---+ +-------+-------+
/// |Int| |Float| |Array| |Tuple|   |FunType| |DataTypeWrapper|
/// +---+ +-----+ +-----+ +-----+   +-------+ +---------------+
///
/// RISE types are divided into three categories that all inherit from
/// mlir::Type:
///    Data types: include Int, Float, Array and Tuple types.
///
///    Natural numbers: Used for tracking the length of the array in the type.
///
///    Rise types: Every RISE value has this type which is either a FunType,
///                representing a RISE function, or a DataTypeWrapper, wrapping
///                a DataType.
///
/// These types follow the type system of the Rise language, but exclude -- for
/// now -- type variables modelled as dependent function types. This type system
/// prevents function types (which are a sub-type of RiseType) to be treated
/// like DataTypes and, for example, be stored in an array.
///

class DataType : public Type::TypeBase<DataType, Type> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == RiseTypeKind::RISE_DATATYPE;
  }

  /// This method is used to get an instance of DataType.
  static DataType get(mlir::MLIRContext *context) {
    return Base::get(context, RiseTypeKind::RISE_DATATYPE);
  }
};

class Nat : public mlir::Type::TypeBase<Nat, Type, detail::RiseNatStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_NAT; }

  /// This method is used to get an instance of Nat.
  static Nat get(mlir::MLIRContext *context, int intValue);

  int getIntValue();
};

class RiseType : public Type::TypeBase<RiseType, Type> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_TYPE; }

  static RiseType get(mlir::MLIRContext *context) {
    return Base::get(context, RiseTypeKind::RISE_TYPE);
  }
};

/// We will experiment with adopting the Integer types of the Standard dialect
class Int : public Type::TypeBase<Int, DataType> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_INT; }

  /// This method is used to get an instance of Int.
  static Int get(mlir::MLIRContext *context) {
    return Base::get(context, RiseTypeKind::RISE_INT);
  }
};

/// We will experiment with adopting the Float types of the Standard dialect
class Float : public Type::TypeBase<Float, DataType> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_FLOAT; }

  /// This method is used to get an instance of Float.
  static Float get(mlir::MLIRContext *context) {
    return Base::get(context, RiseTypeKind::RISE_FLOAT);
  }
};

class ArrayType : public mlir::Type::TypeBase<ArrayType, DataType,
                                              detail::ArrayTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  static mlir::LogicalResult
  verifyConstructionInvariants(llvm::Optional<mlir::Location> loc,
                               mlir::MLIRContext *context, Nat size,
                               DataType elementType);

  /// This method is used to get an instance of ArrayType.
  static ArrayType get(mlir::MLIRContext *context, Nat size,
                       DataType elementType);

  /// Support method to enable LLVM-style RTTI type casting.
  static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_ARRAY; }

  Nat getSize();
  DataType getElementType();
};

class Tuple : public mlir::Type::TypeBase<Tuple, DataType,
                                          detail::RiseTupleTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == RiseTypeKind::RISE_TUPLE; }

  /// This method is used to get an instance of Tuple.
  static Tuple get(mlir::MLIRContext *context, DataType first, DataType second);

  DataType getFirst();
  DataType getSecond();
};

class FunType
    : public Type::TypeBase<FunType, RiseType, detail::RiseFunTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast
  static bool kindof(unsigned kind) {
    return kind == RiseTypeKind::RISE_FUNTYPE;
  }

  /// This method is used to get an instance of FunType
  static FunType get(mlir::MLIRContext *context, RiseType input,
                     RiseType output);

  RiseType getInput();
  RiseType getOutput();
};

class DataTypeWrapper
    : public Type::TypeBase<DataTypeWrapper, RiseType,
                            detail::RiseDataTypeWrapperStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == RiseTypeKind::RISE_DATATYPE_WRAPPER;
  }
  static bool basetype(unsigned kind) {
    return kind == RiseTypeKind::RISE_TYPE;
  }

  /// This method is used to get an instance of DataTypeWrapper.
  static DataTypeWrapper get(mlir::MLIRContext *context, DataType data);

  DataType getDataType();
};

} // end namespace rise
} // end namespace mlir

#endif // MLIR_RISE_TYPES_H
