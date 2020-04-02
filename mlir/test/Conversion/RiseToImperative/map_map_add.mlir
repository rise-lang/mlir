// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e mapMapId -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=MAP_MAP_ADD

func @print_memref_f32(memref<*xf32>)
func @rise_fun(memref<4x4xf32>)
func @mapMapId() {

    rise.fun "rise_fun" (%outArg:memref<4x4xf32>) {
        %array2D = rise.literal #rise.lit<array<4.4, !rise.float, [[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]]>>
        %doubleFun = rise.lambda (%summand) : !rise.fun<data<float> -> data<float>> {
            %addFun = rise.add #rise.float
            %doubled = rise.apply %addFun, %summand, %summand //: !rise.fun<data<float> -> fun<data<float> -> data<float>>>, %summand, %summand
            rise.return %doubled : !rise.data<float>
        }
        %map1 = rise.mapSeq #rise.nat<4> #rise.array<4, !rise.float> #rise.array<4, !rise.float>

        %mapInnerLambda = rise.lambda (%arraySlice) : !rise.fun<data<array<4, float>> -> data<array<4, float>>> {
           %map2 = rise.mapSeq #rise.nat<4> #rise.float #rise.float
           %res = rise.apply %map2, %doubleFun, %arraySlice
           rise.return %res : !rise.data<array<4, float>>
        }
        %res = rise.apply %map1, %mapInnerLambda, %array2D

        rise.return %res: !rise.data<array<4, array<4, float>>>
    }

    //prepare output Array
    %outputArray = alloc() : memref<4x4xf32>
    call @rise_fun(%outputArray) : (memref<4x4xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<4x4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// MAP_MAP_ADD: Unranked Memref rank = 2 descriptor@ = {{.*}}
// MAP_MAP_ADD: Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
// MAP_MAP_ADD: {{[[10,10,10,10],}}
// MAP_MAP_ADD: {{[10,10,10,10],}}
// MAP_MAP_ADD: {{[10,10,10,10],}}
// MAP_MAP_ADD: {{[10,10,10,10]]}}
