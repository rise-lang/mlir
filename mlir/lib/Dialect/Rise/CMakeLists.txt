add_llvm_library(MLIRRise
        RiseRegistration.cpp
        IR/Attributes.cpp
        IR/Ops.cpp
        IR/RiseDialect.cpp
        IR/Types.cpp

        ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rise
        )
add_dependencies(MLIRRise MLIRRiseOpsIncGen MLIRCallOpInterfacesIncGen LLVMSupport)
target_link_libraries(MLIRRise LLVMSupport)
