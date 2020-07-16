set(LLVM_LINK_COMPONENTS
  Core
  Support
  )
add_llvm_executable(rise_highlevel_test
  rise_highlevel_test.cpp
)

llvm_update_compile_flags(rise_highlevel_test)

target_link_libraries(rise_highlevel_test
  PRIVATE
  MLIRAffineOps
  MLIRAffineEDSC
  MLIREDSC
  MLIRIR
  MLIRLinalgEDSC
  MLIRLinalgOps
  MLIRRise
  MLIRRiseEDSC
  MLIRSCF
  MLIRStandardOps
  MLIRTransforms
  MLIRVector
  )

target_include_directories(rise_highlevel_test PRIVATE ..)
