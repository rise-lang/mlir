add_mlir_dialect(Rise Rise)
# usually this should have been sufficient, as the TableGen target is created automatically.
# However, this does not work correctly
set(LLVM_TARGET_DEFINITIONS Rise.td)
mlir_tablegen(Rise.h.inc -gen-op-decls)
mlir_tablegen(Rise.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRRiseOpsIncGen)