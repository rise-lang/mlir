// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-linalg-to-std -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e mapMapId -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=3D_MAP_ADD

func private @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<4x4x4xf32>, %inArg:memref<4x4x4xf32>) {
    rise.lowering_unit {
        %array3D = rise.in %inArg : !rise.array<4, array<4, array<4, scalar<f32>>>>
        %doubleFun = rise.lambda (%summand : !rise.scalar<f32>) -> !rise.scalar<f32> {
            %result = rise.embed(%summand) {
                   %result = addf %summand, %summand : f32
                   rise.return %result : f32
            } : !rise.scalar<f32>
            rise.return %result : !rise.scalar<f32>
        }
        %map1 = rise.mapSeq #rise.nat<4> #rise.array<4, array<4, scalar<f32>>> #rise.array<4, array<4, scalar<f32>>>
        %mapInnerLambda_1 = rise.lambda (%arraySlice_1 : !rise.array<4, array<4, scalar<f32>>>) -> !rise.array<4, array<4, scalar<f32>>> {
            %map2 = rise.mapSeq #rise.nat<4> #rise.array<4, scalar<f32>> #rise.array<4, scalar<f32>>
            %mapInnerLambda_2 = rise.lambda (%arraySlice_2 : !rise.array<4, scalar<f32>>) -> !rise.array<4, scalar<f32>> {
                %map3 = rise.mapSeq #rise.nat<4> #rise.scalar<f32> #rise.scalar<f32>
                %res = rise.apply %map3, %doubleFun, %arraySlice_2
                rise.return %res : !rise.array<4, scalar<f32>>
            }
           %res = rise.apply %map2, %mapInnerLambda_2, %arraySlice_1
           rise.return %res : !rise.array<4, array<4, scalar<f32>>>
        }
        %res = rise.apply %map1, %mapInnerLambda_1, %array3D
        rise.out %outArg <- %res
        rise.return
    }
    return
}
func @mapMapId() {
    //prepare output Array
    %outputArray = memref.alloc() : memref<4x4x4xf32>

    %inputArray = memref.alloc() : memref<4x4x4xf32>
    %cst = constant 5.0 : f32
    linalg.fill(%inputArray, %cst) : memref<4x4x4xf32>, f32

    call @rise_fun(%outputArray, %inputArray) : (memref<4x4x4xf32>, memref<4x4x4xf32>) -> ()

    %print_me = memref.cast %outputArray : memref<4x4x4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// 3D_MAP_ADD: Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [4, 4, 4] strides = [16, 4, 1] data =
// 3D_MAP_ADD: {{[[[10,    10,    10,    10],}}
// 3D_MAP_ADD:     [10,    10,    10,    10],
// 3D_MAP_ADD:     [10,    10,    10,    10],
// 3D_MAP_ADD:     [10,    10,    10,    10]],
// 3D_MAP_ADD:  {{[[10,    10,    10,    10],}}
// 3D_MAP_ADD:     [10,    10,    10,    10],
// 3D_MAP_ADD:     [10,    10,    10,    10],
// 3D_MAP_ADD:     [10,    10,    10,    10]],
// 3D_MAP_ADD:  {{[[10,    10,    10,    10],}}
// 3D_MAP_ADD:     [10,    10,    10,    10],
// 3D_MAP_ADD:     [10,    10,    10,    10],
// 3D_MAP_ADD:     [10,    10,    10,    10]],
// 3D_MAP_ADD:  {{[[10,    10,    10,    10],}}
// 3D_MAP_ADD:     [10,    10,    10,    10],
// 3D_MAP_ADD:     [10,    10,    10,    10],
// 3D_MAP_ADD:     [10,    10,    10,    10]]]