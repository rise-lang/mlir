add_mlir_conversion_library(MLIRRiseToImperative
  ConvertRiseToImperative.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/RiseToImperative

  DEPENDS
  MLIRConversionPassIncGen

  MLIRRise
  MLIRTransforms
  MLIRPass
  MLIRTransforms
  MLIRTranslation
)
