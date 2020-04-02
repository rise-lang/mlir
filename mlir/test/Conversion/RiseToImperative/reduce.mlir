// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e simple_reduction -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_1D_REDUCTION
func @print_memref_f32(memref<*xf32>)
func @rise_fun(memref<1xf32>)
func @simple_reduction() {

    rise.fun "rise_fun" (%outArg:memref<1xf32>) {
        %array0 = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>

        %reductionAdd = rise.lambda (%summand0, %summand1) : !rise.fun<data<float> -> fun<data<float> -> data<float>>> {
            %addFun = rise.add #rise.float
            %doubled = rise.apply %addFun, %summand0, %summand1
            rise.return %doubled : !rise.data<float>
        }
        %initializer = rise.literal #rise.lit<float<0>>
        %reduce4Ints = rise.reduceSeq #rise.nat<4> #rise.float #rise.float
        %result = rise.apply %reduce4Ints, %reductionAdd, %initializer, %array0

        rise.return %result : !rise.data<float>
    }

    //prepare output Array
    %outputArray = alloc() : memref<1xf32>
    call @rise_fun(%outputArray) : (memref<1xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<1xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// SIMPLE_1D_REDUCTION: Unranked Memref rank = 1 descriptor@ = {{.*}}
// SIMPLE_1D_REDUCTION: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// SIMPLE_1D_REDUCTION: [20]
