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

#include "mlir/Dialect/Rise/IR/Types.h"
#include "mlir/Dialect/Rise/IR/TypeDetail.h"
#include "mlir/Dialect/Rise/IR/Dialect.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/STLExtras.h"
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


//===----------------------------------------------------------------------===//
// DataTypeWrapper
//===----------------------------------------------------------------------===//

DataType DataTypeWrapper::getDataType() {
    return getImpl()->data;
}

DataTypeWrapper DataTypeWrapper::get(mlir::MLIRContext *context, DataType data) {
    return Base::get(context, RiseTypeKind::RISE_DATATYPE_WRAPPER, data);
}

//===----------------------------------------------------------------------===//
// Nat
//===----------------------------------------------------------------------===//

int Nat::getIntValue() {
    return getImpl()->intValue;
}

Nat Nat::get(mlir::MLIRContext *context, int intValue) {
    return Base::get(context, RiseTypeKind::RISE_NAT, intValue);
}

//===----------------------------------------------------------------------===//
// FunType
//===----------------------------------------------------------------------===//

FunType FunType::get(mlir::MLIRContext *context, RiseType input, RiseType output) {
    return Base::get(context, RiseTypeKind::RISE_FUNTYPE, input, output);
}

RiseType FunType::getInput() {
    return getImpl()->input;
}

RiseType FunType::getOutput() {
    return getImpl()->output;
}

//===----------------------------------------------------------------------===//
// TupleType
//===----------------------------------------------------------------------===//
Tuple rise::Tuple::get(mlir::MLIRContext *context, DataType first, DataType second) {
    return Base::get(context, RiseTypeKind::RISE_TUPLE, first, second);
}

DataType rise::Tuple::getFirst() {
    return getImpl()->getFirst();
}
DataType rise::Tuple::getSecond() {
    return getImpl()->getSecond();
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

Nat ArrayType::getSize() { return getImpl()->getSize(); }

DataType ArrayType::getElementType() {
    return getImpl()->getElementType();
}

ArrayType ArrayType::get(mlir::MLIRContext *context,
                                 Nat size, DataType elementType) {
    return Base::get(context, RiseTypeKind::RISE_ARRAY, size, elementType);
}

mlir::LogicalResult ArrayType::verifyConstructionInvariants(llvm::Optional<mlir::Location> loc,
                                                             mlir::MLIRContext *context,
                                                             Nat size, DataType elementType) {
    ///For some reason this method is called without a valid location in StorageUniquerSupport
    ///Hence we can not provide proper location information on error

    if (!(size.getIntValue() > 0)) {
        if (loc) {
            emitError(loc.getValue(), "ArrayType must have a size of at least 1");
        } else {
            emitError(UnknownLoc::get(context), "ArrayType must have a size of at least 1");
        }
        return mlir::failure();
    }

    return mlir::success();
}

} //end namespace rise
} //end namespace mlir