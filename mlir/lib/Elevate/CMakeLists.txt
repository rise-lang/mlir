set(CMAKE_CXX_STANDARD 17)
set(LLVM_OPTIONAL_SOURCES
  core.cpp
  traversal.cpp
  algorithmic.cpp
  predicates.cpp
  )
add_mlir_library(MLIRElevate
  core.cpp
  traversal.cpp
  algorithmic.cpp
  predicates.cpp
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Elevate
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rise/Elevate2
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Elevate2

  )
#add_llvm_tool(elevate
#        core.cpp)
