add_llvm_library(MLIRRiseToImperative
  ConvertRiseToImperative.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/RiseToImperative
)
add_dependencies(
  MLIRRiseToImperative

  MLIRRise
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
target_link_libraries(
  MLIRRiseToImperative

  MLIRRise
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
