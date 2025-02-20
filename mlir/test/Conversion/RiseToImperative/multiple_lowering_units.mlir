// RUN: mlir-opt %s -convert-rise-to-imperative -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e multiple -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=MULTIPLE

func private @print_memref_f32(memref<*xf32>)

func @rise_fun(%outArg: memref<6xf32>, %in: memref<6xf32>) {
    rise.lowering_unit {
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
        rise.return
    }
    rise.lowering_unit {
        %array = rise.in %outArg : !rise.array<6, scalar<f32>>

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
        rise.return
    }
    return
}

func @multiple() {
    //prepare output Array
    %outputArray = memref.alloc() : memref<6xf32>

    %inputArray = memref.alloc() : memref<6xf32>
    %cst = constant 5.0 : f32
    linalg.fill(%cst, %inputArray) : f32, memref<6xf32>

    call @rise_fun(%outputArray, %inputArray) : (memref<6xf32>, memref<6xf32>) -> ()

    %print_me = memref.cast %outputArray : memref<6xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// MULTIPLE: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [6] strides = [1] data =
// MULTIPLE: [20, 20, 20, 20, 20, 20]

