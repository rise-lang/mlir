#set(LIBS
#
#  MLIRIR
#  MLIRStandardOps
#  )
#
#add_llvm_library(MLIRLinalgOps
#  LinalgOps.cpp
#  LinalgTypes.cpp
#
#  ADDITIONAL_HEADER_DIRS
#  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg
#  DEPENDS
#  intrinsics_gen
#  )
#
#add_dependencies(MLIRLinalgOps
#
#  ${LIBS}
#  MLIRLinalgOpsIncGen
#  MLIRLinalgStructuredOpsIncGen
#  )
#target_link_libraries(MLIRLinalgOps ${LIBS})
#
#
#
#
#
#
add_llvm_library(MLIRRise
  Attributes.cpp
  Ops.cpp
  RiseDialect.cpp
  Types.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rise
  DEPENDS
  intrinsics_gen
  )
add_dependencies(MLIRRise MLIRRiseIncGen MLIRIR MLIRCallOpInterfacesIncGen LLVMSupport)
target_link_libraries(MLIRRise MLIRIR LLVMSupport)
