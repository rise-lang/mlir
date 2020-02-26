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

#ifndef MLIR_RISE_ATTRIBUTES_H
#define MLIR_RISE_ATTRIBUTES_H

#include "AttributeDetail.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace rise {

namespace detail {
struct DataTypeAttributeStorage;
struct NatAttributeStorage;
struct LiteralAttributeStorage;
} // namespace detail

/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum RiseAttributeKind {
  // The enum starts at the range reserved for this dialect.
  RISE_ATTR = mlir::Attribute::FIRST_RISE_ATTR,
  NAT_ATTR,
  LITERAL_ATTR,
  DATATYPE_ATTR,
};

/// Rise LiteralAttr is used to pass information about type and value of
/// a literal to the RISE literal operation.
///
/// This format is not the one used in the paper and will change to it soon.
/// current Format:
///         #rise.int<42>
///         #rise.array<2, rise.int, [1,2]>
///         #rise.array<2.3, !rise.int, [[1,2,3],[4,5,6]]>
class LiteralAttr
    : public Attribute::AttrBase<LiteralAttr, Attribute,
                                 detail::LiteralAttributeStorage> {
public:
  using Base::Base;

  static LiteralAttr get(MLIRContext *context, DataType type,
                         std::string value);

  std::string getValue() const;
  DataType getType() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == RiseAttributeKind::LITERAL_ATTR;
  }
};

/// RISE DataTypeAttr is used to specialize certain functions to a DataType
/// e.g. rise.add #rise.int returns an addition operation for integers.
class DataTypeAttr
    : public Attribute::AttrBase<DataTypeAttr, Attribute,
                                 detail::DataTypeAttributeStorage> {
public:
  using Base::Base;

  static DataTypeAttr get(MLIRContext *context, DataType value);

  DataType getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == RiseAttributeKind::DATATYPE_ATTR;
  }
};

/// RISE NatAttr is used to specify the number of elements of Array.
class NatAttr : public Attribute::AttrBase<NatAttr, Attribute,
                                           detail::NatAttributeStorage> {
public:
  using Base::Base;

  static NatAttr get(MLIRContext *context, Nat value);

  Nat getValue() const;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool kindof(unsigned kind) {
    return kind == RiseAttributeKind::NAT_ATTR;
  }
};

} // namespace rise
}
#endif // MLIR_RISE_ATTRIBUTES_H
;      // namespace mlir