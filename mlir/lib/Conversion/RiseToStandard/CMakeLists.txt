add_llvm_library(MLIRRiseToStandard
  ConvertRiseToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/RiseToStandard
)
add_dependencies(
  MLIRRiseToStandard

  MLIRRise
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
target_link_libraries(
  MLIRRiseToStandard

  MLIRRise
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
