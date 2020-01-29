add_llvm_library(RISE
        RiseRegistration.cpp
        IR/Attributes.cpp
        IR/Ops.cpp
        IR/RiseDialect.cpp
        IR/Types.cpp

        ADDITIONAL_HEADER_DIRS
        ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Rise
        )
add_dependencies(RISE MLIRRiseOpsIncGen MLIRCallOpInterfacesIncGen LLVMSupport)
target_link_libraries(RISE LLVMSupport)
