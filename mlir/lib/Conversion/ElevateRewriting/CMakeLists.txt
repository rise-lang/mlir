add_mlir_conversion_library(MLIRElevateRewriting
  ElevateRewriting.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ElevateRewriting
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Elevate

  DEPENDS
  MLIRConversionPassIncGen
  MLIRElevate
  MLIRRise
  MLIRTransforms
  MLIRPass
  MLIRTransforms
  MLIRTranslation
)
