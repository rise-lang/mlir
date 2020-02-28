// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e simple_reduction -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_1D_REDUCTION
func @print_memref_f32(memref<*xf32>)
func @simple_reduction() {

    %res = rise.fun {
        //Array
        %array0 = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>

        //Reduction
        %addFun = rise.add #rise.float
        %initializer = rise.literal #rise.lit<float<0>>
        %reduce4Ints = rise.reduce #rise.nat<4> #rise.float #rise.float
        %result = rise.apply %reduce4Ints, %addFun, %initializer, %array0

        rise.return %result : !rise.data<float>
    }

    %print_me = memref_cast %res : memref<1xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()

    return
}
// SIMPLE_1D_REDUCTION: Unranked Memref rank = 1 descriptor@ = {{.*}}
// SIMPLE_1D_REDUCTION: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// SIMPLE_1D_REDUCTION: [20]
