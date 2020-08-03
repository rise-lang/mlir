// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-linalg-to-std -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e simple_snd -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_SND

func @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<4xf32>, %inArg0:memref<4xf32>, %inArg1:memref<4xf32>) {
    rise.lowering_unit {
        %array0 = rise.in %inArg0 : !rise.array<4, scalar<f32>>
        %array1 = rise.in %inArg1 : !rise.array<4, scalar<f32>>

        %zipFun = rise.zip #rise.nat<4> #rise.scalar<f32> #rise.scalar<f32>
        %zipped = rise.apply %zipFun, %array0, %array1

        %projectToFirst = rise.lambda (%floatTuple : !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32> {
            %sndFun = rise.snd #rise.scalar<f32> #rise.scalar<f32>
            %snd = rise.apply %sndFun, %floatTuple
            rise.return %snd : !rise.scalar<f32>
        }

        %mapFun = rise.mapSeq #rise.nat<4> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
        %fstArray = rise.apply %mapFun, %projectToFirst, %zipped
        rise.out %outArg <- %fstArray
        rise.return
    }
    return
}

func @simple_snd() {
    //prepare output Array
    %outputArray = alloc() : memref<4xf32>

    %inArg0 = alloc() : memref<4xf32>
    %cst_5 = constant 5.0 : f32
    linalg.fill(%inArg0, %cst_5) : memref<4xf32>, f32

    %inArg1 = alloc() : memref<4xf32>
    %cst_10 = constant 10.0 : f32
    linalg.fill(%inArg1, %cst_10) : memref<4xf32>, f32

    call @rise_fun(%outputArray, %inArg0, %inArg1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// SIMPLE_SND: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// SIMPLE_SND: [10, 10, 10, 10]