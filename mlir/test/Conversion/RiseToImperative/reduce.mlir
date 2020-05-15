// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e simple_reduction -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_1D_REDUCTION
func @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<1xf32>, %inArg:memref<1024xf32>) {
    %array0 = rise.in %inArg : !rise.array<1024, scalar<f32>>

    %reductionAdd = rise.lambda (%summand0, %summand1) : !rise.fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>> {
        %summand0_unwrapped = rise.unwrap %summand0
        %summand1_unwrapped = rise.unwrap %summand1
        %result = addf %summand0_unwrapped, %summand1_unwrapped : f32
//            %result = constant 2.0 : f32
        %result_wrapped = rise.wrap %result
        rise.return %result_wrapped : !rise.scalar<f32>
    }
    %initializer = rise.literal #rise.lit<0.0>
    %reduce4Ints = rise.reduceSeq #rise.nat<1024> #rise.scalar<f32> #rise.scalar<f32>
    %result = rise.apply %reduce4Ints, %reductionAdd, %initializer, %array0

    return
}

func @simple_reduction() {

    //prepare output Array
    %outputArray = alloc() : memref<1xf32>
    %cst_0 = constant 0.0 : f32
    linalg.fill(%outputArray, %cst_0) : memref<1xf32>, f32

    %inArg0 = alloc() : memref<1024xf32>
    %cst_5 = constant 5.0 : f32
    linalg.fill(%inArg0, %cst_5) : memref<1024xf32>, f32

    call @rise_fun(%outputArray, %inArg0) : (memref<1xf32>, memref<1024xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<1xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// SIMPLE_1D_REDUCTION: Unranked Memref rank = 1 descriptor@ = {{.*}}
// SIMPLE_1D_REDUCTION: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// SIMPLE_1D_REDUCTION: [5120]
