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

#include "mlir/Dialect/Rise/IR/Attributes.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace rise {

//===----------------------------------------------------------------------===//
// LiteralAttr
//===----------------------------------------------------------------------===//

LiteralAttr LiteralAttr::get(MLIRContext *context, DataType type,
                             std::string value) {
  return Base::get(context, RiseAttributeKind::LITERAL_ATTR, type, value);
}
DataType LiteralAttr::getType() const { return getImpl()->type; }

std::string LiteralAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// DataTypeAttr
//===----------------------------------------------------------------------===//

DataTypeAttr DataTypeAttr::get(MLIRContext *context, DataType value) {
  return Base::get(context, RiseAttributeKind::DATATYPE_ATTR, value);
}
DataType DataTypeAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// NatAttr
//===----------------------------------------------------------------------===//

NatAttr NatAttr::get(MLIRContext *context, Nat value) {
  return Base::get(context, RiseAttributeKind::NAT_ATTR, value);
}
Nat NatAttr::getValue() const { return getImpl()->value; }

} // end namespace rise
} // end namespace mlir