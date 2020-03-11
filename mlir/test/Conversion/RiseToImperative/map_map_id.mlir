// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e mapMapId -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=MAP_MAP_ID

func @print_memref_f32(memref<*xf32>)
func @mapMapId() {

    %res = rise.fun {
        %array2D = rise.literal #rise.lit<array<4.4, !rise.float, [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]>>
        %doubleFun = rise.lambda (%summand) : !rise.fun<data<float> -> data<float>> {
            %addFun = rise.add #rise.float
            %doubled = rise.apply %addFun, %summand, %summand //: !rise.fun<data<float> -> fun<data<float> -> data<float>>>, %summand, %summand
            rise.return %doubled : !rise.data<float>
        }
        %map1 = rise.map #rise.nat<4> #rise.array<4, !rise.float> #rise.array<4, !rise.float>
        %map2 = rise.map #rise.nat<4> #rise.float #rise.float

//  partial application
// lowering assumption here is false think more about when the assumptions breaks
// -> it breaks when something in the middle of the apply chain has to be applied to before the chain can continue.
        %mapInner = rise.apply %map2, %doubleFun //: !rise.fun<fun<data<float> -> data<float>> -> fun<data<array<4, float>> -> data<array<4, float>>>>, %doubleFun
        %map2D = rise.apply %map1, %mapInner //: !rise.fun<fun<data<array<4, float>> -> data<array<4, float>>> -> fun<data<array<4, array<4, float>>> -> data<array<4, array<4, float>>>>>, %mapInner

        %res = rise.apply %map2D, %array2D
//        %res = rise.apply %map1, %map2, %doubleFun, %array2D
        rise.return %res: !rise.data<array<4, array<4, float>>>
    } : () -> memref<4x4xf32>

    %print_me = memref_cast %res : memref<4x4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// MAP_MAP_ID: Unranked Memref rank = 1 descriptor@ = {{.*}}
// MAP_MAP_ID: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// MAP_MAP_ID: [10, 10, 10, 10]

