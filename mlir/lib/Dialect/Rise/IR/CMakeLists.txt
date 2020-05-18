add_mlir_dialect_library(MLIRRise
  Attributes.cpp
  Ops.cpp
  RiseDialect.cpp
  Types.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rise

  DEPENDS
  MLIRRiseIncGen
  MLIRRisePassIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRStandardOps
  )
