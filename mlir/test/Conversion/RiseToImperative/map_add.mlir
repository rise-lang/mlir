// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e array_times_2 -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=ARRAY_TIMES_2

func @print_memref_f32(memref<*xf32>)
//func @rise_fun(%outArg: memref<4xf32>, %in: memref<4xf32>) {
//    // This way we dont have to handle moving the block arguments anymore.
//    %array = rise.in %in : !rise.array<4, scalar<f32>>
//    %doubleFun = rise.lambda (%summand) : !rise.fun<scalar<f32> -> scalar<f32>> {
//        %summandUnwrapped = rise.unwrap %summand
//        %doubled = addf %summandUnwrapped, %summandUnwrapped : f32    // This section can contain arbitrary operations using the %summandUnwrapped. This is opaque to RISE
//        %doubledWrapped = rise.wrap %doubled
//        rise.return %doubledWrapped : !rise.scalar<f32>
//    }
//    %map4IntsToInts = rise.mapSeq {to = "loop"} #rise.nat<4> #rise.scalar<f32> #rise.scalar<f32>
//    %doubledArray = rise.apply %map4IntsToInts, %doubleFun, %array
//
//    return
//}
func @rise_fun(%outArg: memref<4xf32>, %in: memref<4xf32>) {
    // This way we dont have to handle moving the block arguments anymore.
    %array = rise.in %in : !rise.array<4, scalar<f32>>
    %doubleFun = rise.lambda (%summand : !rise.scalar<f32>) -> !rise.scalar<f32> {
        %result = rise.embed(%summand) {
            %doubled = addf %summand, %summand : f32
            rise.return %doubled : f32
        }
        rise.return %result : !rise.scalar<f32>
    }
    %map = rise.mapSeq {to = "loop"} #rise.nat<4> #rise.scalar<f32> #rise.scalar<f32>
    %doubledArray = rise.apply %map, %doubleFun, %array
    rise.out %outArg <- %doubledArray
    return
}

func @array_times_2() {
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
// ARRAY_TIMES_2: Unranked Memref rank = 1 descriptor@ = {{.*}}
// ARRAY_TIMES_2: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// ARRAY_TIMES_2: [10, 10, 10, 10]

