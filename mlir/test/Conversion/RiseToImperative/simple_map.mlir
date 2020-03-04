// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e simple_map_example -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_ARRAY_DOUBLING
func @print_memref_f32(memref<*xf32>)
func @simple_map_example() {

    %res = rise.fun {
        %array = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>
        %doubleFun = rise.lambda (%summand) : !rise.fun<data<float> -> data<float>> {
            %addFun = rise.add #rise.float
            %doubled = rise.apply %addFun, %summand, %summand
            rise.return %doubled : !rise.data<float>
        }
        %map4IntsToInts = rise.map #rise.nat<4> #rise.float #rise.float
//        %mapDoubleFun = rise.apply %map4IntsToInts, %doubleFun
//        %doubledArray = rise.apply %mapDoubleFun, %array
        %doubledArray = rise.apply %map4IntsToInts, %doubleFun, %array

        rise.return %doubledArray : !rise.data<array<4, float>>
    } : () -> memref<4xf32>

    %print_me = memref_cast %res : memref<4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// SIMPLE_ARRAY_DOUBLING: Unranked Memref rank = 1 descriptor@ = {{.*}}
// SIMPLE_ARRAY_DOUBLING: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// SIMPLE_ARRAY_DOUBLING: [10, 10, 10, 10]
