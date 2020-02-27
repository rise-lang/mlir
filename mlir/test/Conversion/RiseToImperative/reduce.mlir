// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e main -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=PRINT-0D
func @main() {

    rise.module { // to be rise interface
        //Array
        %array0 = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>

        //Reduction
        %addFun = rise.add #rise.float
        %initializer = rise.literal #rise.lit<float<0>>
        %reduce4Ints = rise.reduce #rise.nat<4> #rise.float #rise.float
        %result = rise.apply %reduce4Ints, %addFun, %initializer, %array0

        rise.return %result : !rise.data<float>
    }

    return
}