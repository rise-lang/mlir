add_mlir_dialect_library(MLIRRiseEDSC
  Builders.cpp
  HighLevel.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rise

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRRise
  MLIRAffineEDSC
  MLIRSCF
  MLIRStandardOps
  )
