add_mlir_conversion_library(MLIRRiseBuilderTest
  RiseBuilderTest.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/RiseBuilderTest

  DEPENDS
  MLIRConversionPassIncGen

  MLIRRise
  MLIRTransforms
  MLIRPass
  MLIRTransforms
  MLIRTranslation
)
