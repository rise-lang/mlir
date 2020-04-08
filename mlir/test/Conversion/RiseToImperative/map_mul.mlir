// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e array_times_5 -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=ARRAY_TIMES_5

func @print_memref_f32(memref<*xf32>)
func @rise_fun(memref<4xf32>, memref<4xf32>)
func @array_times_5() {

    rise.fun "rise_fun" (%outArg:memref<4xf32>, %in:memref<4xf32>) {
        %array = rise.in %in : !rise.array<4, scalar<f32>>
        %times5 = rise.lambda (%elem) : !rise.fun<scalar<f32> -> scalar<f32>> {
            %cst5 = rise.literal #rise.lit<5.0>
            %unwrappedLiteral = rise.unwrap %cst5
            %unwrappedElem = rise.unwrap %elem
            %result = mulf %unwrappedElem, %unwrappedLiteral : f32
            %resultWrapped = rise.wrap %result
            rise.return %resultWrapped : !rise.scalar<f32>
        }
        %mapFun = rise.mapSeq #rise.nat<4> #rise.scalar<f32> #rise.scalar<f32>
        %multipliedArray = rise.apply %mapFun, %times5, %array

        rise.return %multipliedArray : !rise.array<4, scalar<f32>>
    }

    //prepare output Array
    %outputArray = alloc() : memref<4xf32>

    %inputArray = alloc() : memref<4xf32>
    %cst = constant 5.0 : f32
    linalg.fill(%inputArray, %cst) : memref<4xf32>, f32

    call @rise_fun(%outputArray, %inputArray) : (memref<4xf32>, memref<4xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// ARRAY_TIMES_5: Unranked Memref rank = 1 descriptor@ = {{.*}}
// ARRAY_TIMES_5: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// ARRAY_TIMES_5: [25, 25, 25, 25]
