// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e simple_fst -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_FST

func @print_memref_f32(memref<*xf32>)
func @rise_fun(memref<4xf32>)
func @simple_fst() {

    rise.fun "rise_fun" (%outArg:memref<4xf32>) {
        %array0 = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>
        %array1 = rise.literal #rise.lit<array<4, !rise.float, [10,10,10,10]>>

        %zipFun = rise.zip #rise.nat<4> #rise.float #rise.float
        %zipped = rise.apply %zipFun, %array0, %array1

        %projectToFirst = rise.lambda (%floatTuple) : !rise.fun<data<tuple<float, float>> -> data<float>> {
            %fstFun = rise.fst #rise.float #rise.float
            %fst = rise.apply %fstFun, %floatTuple
            rise.return %fst : !rise.data<float>
        }

        %mapFun = rise.mapSeq #rise.nat<4> #rise.tuple<float, float> #rise.float
        %fstArray = rise.apply %mapFun, %projectToFirst, %zipped

        rise.return %fstArray : !rise.data<array<4, float>>
    }

    //prepare output Array
    %outputArray = alloc() : memref<4xf32>
    call @rise_fun(%outputArray) : (memref<4xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// SIMPLE_FST: Unranked Memref rank = 1 descriptor@ = {{.*}}
// SIMPLE_FST: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// SIMPLE_FST: [5, 5, 5, 5]

