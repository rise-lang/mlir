// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e fused_dot -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_DOT

func private @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<f32>, %inArg0:memref<1024xf32>, %inArg1:memref<1024xf32>)  {
    rise.lowering_unit {
        //Arrays
        %array0 = rise.in %inArg0 : !rise.array<1024, scalar<f32>>
        %array1 = rise.in %inArg1 : !rise.array<1024, scalar<f32>>

        //Zipping
        %zipFun = rise.zip #rise.nat<1024> #rise.scalar<f32> #rise.scalar<f32>
        %zippedArrays = rise.apply %zipFun, %array0, %array1

        //Reduction
        %reductionLambda = rise.lambda (%tuple : !rise.tuple<scalar<f32>, scalar<f32>>, %acc : !rise.scalar<f32>) -> !rise.scalar<f32> {

            %fstFun = rise.fst #rise.scalar<f32> #rise.scalar<f32>
            %sndFun = rise.snd #rise.scalar<f32> #rise.scalar<f32>

            %fst = rise.apply %fstFun, %tuple
            %snd = rise.apply %sndFun, %tuple

            %result = rise.embed(%fst, %snd, %acc) {
                   %product = mulf %fst, %snd :f32
                   %result = addf %product, %acc : f32
                   rise.return %result : f32
            } : !rise.scalar<f32>

            rise.return %result : !rise.scalar<f32>
        }

        %initializer = rise.literal #rise.lit<0.0>
        %reduceFun = rise.reduceSeq #rise.nat<1024> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
        %result = rise.apply %reduceFun, %reductionLambda, %initializer, %zippedArrays
        rise.out %outArg <- %result
        rise.return
    }
    return
}

func @fused_dot() {
    //prepare output Array
    %outputArray = memref.alloc() : memref<f32>
    %cst_0 = constant 0.0 : f32
    linalg.fill(%cst_0, %outputArray) : f32, memref<f32>

    %inArg0 = memref.alloc() : memref<1024xf32>
    %cst_5 = constant 5.0 : f32
    linalg.fill(%cst_5, %inArg0) : f32, memref<1024xf32>

    %inArg1 = memref.alloc() : memref<1024xf32>
    %cst_10 = constant 5.0 : f32
    linalg.fill(%cst_10, %inArg1) : f32, memref<1024xf32>

    call @rise_fun(%outputArray, %inArg0, %inArg1) : (memref<f32>, memref<1024xf32>, memref<1024xf32>) -> ()

    %print_me = memref.cast %outputArray : memref<f32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// SIMPLE_DOT: Unranked Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
// SIMPLE_DOT: [25600]

