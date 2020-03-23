// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e array_times_5 -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=ARRAY_TIMES_5

func @print_memref_f32(memref<*xf32>)
func @rise_fun(memref<4xf32>)
func @array_times_5() {

    rise.fun "rise_fun" (%outArg:memref<4xf32>) {
        %array = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>
        %times5 = rise.lambda (%in) : !rise.fun<data<float> -> data<float>> {
            %cst5 = rise.literal #rise.lit<float<5>>
            %mulFun = rise.mult #rise.float
            %multiplied = rise.apply %mulFun, %in, %cst5
            rise.return %multiplied : !rise.data<float>
        }
        %mapFun = rise.map #rise.nat<4> #rise.float #rise.float
        %multipliedArray = rise.apply %mapFun, %times5, %array

        rise.return %multipliedArray : !rise.data<array<4, float>>
    }

    //prepare output Array
    %outputArray = alloc() : memref<4xf32>
    call @rise_fun(%outputArray) : (memref<4xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// ARRAY_TIMES_5: Unranked Memref rank = 1 descriptor@ = {{.*}}
// ARRAY_TIMES_5: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// ARRAY_TIMES_5: [25, 25, 25, 25]
