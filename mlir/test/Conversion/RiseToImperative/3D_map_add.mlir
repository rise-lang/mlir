// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-std -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e mapMapId -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=3D_MAP_ADD

func @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<4x4x4xf32>, %inArg:memref<4x4x4xf32>) {
    %array3D = rise.in %inArg : !rise.array<4, array<4, array<4, scalar<f32>>>>
    %doubleFun = rise.lambda (%summand) : !rise.fun<scalar<f32> -> scalar<f32>> {
        %summand_unwrapped = rise.unwrap %fst
        %result = addf %summand_unwrapped, %summand_unwrapped :f32
        %result_wrapped = rise.wrap %result
        rise.return %result_wrapped : !rise.scalar<f32>
    }
    %map1 = rise.mapSeq #rise.nat<4> #rise.array<4, array<4, scalar<f32>>> #rise.array<4, array<4, scalar<f32>>>
    %mapInnerLambda_1 = rise.lambda (%arraySlice_1) : !rise.fun<array<4, array<4, scalar<f32>>> -> array<4, array<4, scalar<f32>>>> {
        %map2 = rise.mapSeq #rise.nat<4> #rise.array<4, scalar<f32>> #rise.array<4, scalar<f32>>
        %mapInnerLambda_2 = rise.lambda (%arraySlice_2) : !rise.fun<array<4, scalar<f32>> -> array<4, scalar<f32>>> {
            %map3 = rise.mapSeq #rise.nat<4> #rise.scalar<f32> #rise.scalar<f32>
            %res = rise.apply %map3, %doubleFun, %arraySlice_2
            rise.return %res : !rise.array<4, scalar<f32>>
        }
       %res = rise.apply %map2, %mapInnerLambda_2, %arraySlice_1
       rise.return %res : !rise.array<4, array<4, scalar<f32>>>
    }
    %res = rise.apply %map1, %mapInnerLambda_1, %array3D
    return
}
func @mapMapId() {
    //prepare output Array
    %outputArray = alloc() : memref<4x4x4xf32>

    %inputArray = alloc() : memref<4x4x4xf32>
    %cst = constant 5.0 : f32
    linalg.fill(%inputArray, %cst) : memref<4x4x4xf32>, f32

    call @rise_fun(%outputArray, %inputArray) : (memref<4x4x4xf32>, memref<4x4x4xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<4x4x4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// 3D_MAP_ADD: Unranked Memref rank = 3 descriptor@ = {{.*}}
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