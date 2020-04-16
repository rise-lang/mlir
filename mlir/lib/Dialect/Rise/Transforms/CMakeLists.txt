set(LIBS

  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRIR
  MLIRRise
  MLIRLoopOps
  MLIRPass
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRTransformUtils
  MLIRVectorOps
  )

add_llvm_library(MLIRRiseTransforms
  Rewriting.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rise/Transforms
  )

add_dependencies(MLIRRiseTransforms

  ${LIBS}
  )
target_link_libraries(MLIRRiseTransforms ${LIBS})
