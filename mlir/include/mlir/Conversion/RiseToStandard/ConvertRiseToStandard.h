//
// Created by martin on 26/11/2019.
//

#ifndef MLIR_CONVERTRISETOSTANDARD_H
#define MLIR_CONVERTRISETOSTANDARD_H

#include "mlir/Dialect/Rise/IR/Types.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class RiseTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  RiseTypeConverter(MLIRContext *ctx);

  /// Convert types to Standard.
  Type convertType(Type t) override;
  Type convertIntType(rise::Int type);
};
} // namespace mlir

#endif // MLIR_CONVERTRISETOSTANDARD_H
