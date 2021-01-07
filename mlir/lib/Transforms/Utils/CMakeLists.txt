add_mlir_library(MLIRTransformUtils
  DialectConversion.cpp
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  ElevateRewriteDriver.cpp
  InliningUtils.cpp
  LoopFusionUtils.cpp
  LoopUtils.cpp
  RegionUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Elevate2


  DEPENDS
  MLIRStandardOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRLoopAnalysis
  MLIRElevate
  MLIRSCF
  MLIRPass
  MLIRRewrite
  MLIRStandard
  )
