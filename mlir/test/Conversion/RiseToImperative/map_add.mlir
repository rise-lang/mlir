// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e array_times_2 -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=ARRAY_TIMES_2

func @print_memref_f32(memref<*xf32>)

func @rise_fun(%outArg: memref<6xf32>, %in: memref<6xf32>) {
    // This way we dont have to handle moving the block arguments anymore.
    %array = rise.in %in : !rise.array<6, scalar<f32>>

    %doubleFun = rise.lambda (%summand : !rise.scalar<f32>) -> !rise.scalar<f32> {
        %result = rise.embed(%summand) {
            %doubled = addf %summand, %summand : f32
            rise.return %doubled : f32
        } : !rise.scalar<f32>
        rise.return %result : !rise.scalar<f32>
    }
    %map = rise.mapSeq {to = "scf"} #rise.nat<6> #rise.scalar<f32> #rise.scalar<f32>
    %doubledArray = rise.apply %map, %doubleFun, %array
    rise.out %outArg <- %doubledArray
    return
}

func @array_times_2() {
    //prepare output Array
    %outputArray = alloc() : memref<6xf32>

    %inputArray = alloc() : memref<6xf32>
    %cst = constant 5.0 : f32
    linalg.fill(%inputArray, %cst) : memref<6xf32>, f32

    call @rise_fun(%outputArray, %inputArray) : (memref<6xf32>, memref<6xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<6xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// ARRAY_TIMES_2: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [6] strides = [1] data =
// ARRAY_TIMES_2: [10, 10, 10, 10, 10, 10]

