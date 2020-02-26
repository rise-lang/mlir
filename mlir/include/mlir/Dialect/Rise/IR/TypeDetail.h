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

#ifndef MLIR_RISE_TYPEDETAIL_H
#define MLIR_RISE_TYPEDETAIL_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
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
namespace detail {

/// This class holds the implementation of the Rise DataTypeWrapper.
struct RiseDataTypeWrapperStorage : public mlir::TypeStorage {
  RiseDataTypeWrapperStorage(DataType data) : data(data) {}
  /// This defines how we unique this type in the context: Rise DataTypeWrapper
  /// types are unique by the wrapped DataType
  using KeyTy = DataType;

  bool operator==(const KeyTy &key) const { return key == KeyTy(data); }

  static KeyTy getKey(DataType data) { return KeyTy(data); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key.getAsOpaquePointer());
  }

  static RiseDataTypeWrapperStorage *
  construct(mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<RiseDataTypeWrapperStorage>())
        RiseDataTypeWrapperStorage(key);
  }

  DataType data;
};

/// This class holds the implementation of the Rise NatStorage
struct RiseNatStorage : public mlir::TypeStorage {
  RiseNatStorage(int intValue) : intValue(intValue) {}
  /// This defines how we unique this type in the context: Rise DataTypeWrapper
  /// types are unique by the wrapped DataType
  using KeyTy = int;

  bool operator==(const KeyTy &key) const { return key == KeyTy(intValue); }

  static KeyTy getKey(int intValue) { return KeyTy(intValue); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static RiseNatStorage *construct(mlir::TypeStorageAllocator &allocator,
                                   const KeyTy &key) {
    return new (allocator.allocate<RiseNatStorage>()) RiseNatStorage(key);
  }

  int intValue;
};

/// This class holds the implementation of the Rise FunType.
struct RiseFunTypeStorage : public mlir::TypeStorage {
  RiseFunTypeStorage(RiseType input, RiseType output)
      : input(input), output(output) {}
  /// This defines how we unique this type in the context: Two Rise FunTypes are
  /// equal when input and output are equal.
  using KeyTy = std::pair<RiseType, RiseType>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(input, output);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static KeyTy getKey(RiseType input, RiseType output) {
    return KeyTy(input, output);
  }

  static RiseFunTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<RiseFunTypeStorage>())
        RiseFunTypeStorage(key.first, key.second);
  }

  RiseType input;
  RiseType output;
};

/// This class holds the implementation of the Rise Tuple.
struct RiseTupleTypeStorage : public mlir::TypeStorage {
  RiseTupleTypeStorage(DataType first, DataType second)
      : first(first), second(second) {}
  /// This defines how we unique this type in the context: Rise Tuple types are
  /// unique by the types of the two contained elements: "first" and "second"
  using KeyTy = std::pair<DataType, DataType>;

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(first, second);
  }

  static KeyTy getKey(DataType first, DataType second) {
    return KeyTy(first, second);
  }

  static RiseTupleTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    return new (allocator.allocate<RiseTupleTypeStorage>())
        RiseTupleTypeStorage(key.first, key.second);
  }

  DataType getFirst() const { return first; }
  DataType getSecond() const { return second; }

private:
  DataType first;
  DataType second;
};

/// This class holds the implementation of the Rise ArrayType.
struct ArrayTypeStorage : public mlir::TypeStorage {
  ArrayTypeStorage(Nat size, DataType elementType)
      : size(size), elementType(elementType) {}
  /// This defines how we unique this type in the context: Array Types are
  /// unique by size and type of the elements
  using KeyTy = std::pair<Nat, DataType>;

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(size, elementType);
  }

  static KeyTy getKey(Nat size, DataType elementType) {
    return KeyTy(size, elementType);
  }

  static ArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ArrayTypeStorage>())
        ArrayTypeStorage(key.first, key.second);
  }

  Nat getSize() const { return size; }
  DataType getElementType() const { return elementType; }

private:
  Nat size;
  DataType elementType;
};

} // end namespace detail
} // end namespace rise
} // end namespace mlir
#endif // MLIR_RISE_TYPEDETAIL_H
