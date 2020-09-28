set(CMAKE_CXX_STANDARD 17)
add_mlir_library(MLIRElevate
        core.cpp

        ADDITIONAL_HEADER_DIRS
          ${MLIR_MAIN_INCLUDE_DIR}/mlir/Elevate

        )
add_llvm_tool(elevate
        core.cpp)


#add_mlir_dialect_library(MLIRRise
#  Attributes.cpp
#  Ops.cpp
#  RiseDialect.cpp
#  Types.cpp
#
#  ADDITIONAL_HEADER_DIRS
#  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rise
#
#  DEPENDS
#  MLIRRiseIncGen
#  MLIRRisePassIncGen
#
#  LINK_LIBS PUBLIC
#  MLIREDSC
#  MLIRIR
#  MLIRLoopLikeInterface
#  MLIRSideEffectInterfaces
#  MLIRStandardOps
#  )
