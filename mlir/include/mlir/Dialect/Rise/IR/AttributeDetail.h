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
#ifndef MLIR_RISE_ATTRIBUTEDETAIL_H
#define MLIR_RISE_ATTRIBUTEDETAIL_H

#include "Attributes.h"
#include "Types.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir {
namespace rise {
namespace detail {

/// Implementation of the LiteralAttr
struct LiteralAttributeStorage : public mlir::AttributeStorage {
  LiteralAttributeStorage(DataType type, std::string value)
      : type(type), value(value) {}

  // This is intentionally a StringRef, hashing does not work with std::string
  // for some reason
  using KeyTy = std::pair<DataType, std::string>;

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == KeyTy(type, value); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// Construct a new storage instance.
  static LiteralAttributeStorage *
  construct(mlir::AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<LiteralAttributeStorage>())
        LiteralAttributeStorage(key.first, key.second);
  }

  DataType type;
  std::string value;
};

/// Implementation of the DataTypeAttr
struct DataTypeAttributeStorage : public mlir::AttributeStorage {
  DataTypeAttributeStorage(DataType value) : value(value) {}
  using KeyTy = DataType;

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == KeyTy(value); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key.getAsOpaquePointer());
  }

  /// Construct a new storage instance.
  static DataTypeAttributeStorage *
  construct(mlir::AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<DataTypeAttributeStorage>())
        DataTypeAttributeStorage(key);
  }

  DataType value;
};

/// Implementation of the NatAttr
struct NatAttributeStorage : public mlir::AttributeStorage {
  NatAttributeStorage(Nat value) : value(value) {}
  using KeyTy = Nat;

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == KeyTy(value); }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key.getAsOpaquePointer());
  }

  /// Construct a new storage instance.
  static NatAttributeStorage *
  construct(mlir::AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<NatAttributeStorage>())
        NatAttributeStorage(key);
  }

  Nat value; // TODO: This is the culprit for the current errors. Make the rest
             // conform to this. Add a getValue method to Nat
};
} // namespace detail
} // namespace rise
} // namespace mlir

#endif // LLVM_ATTRIBUTEDETAIL_H
