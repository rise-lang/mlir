// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e fused_dot -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_DOT

func @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<1xf32>, %inArg0:memref<1024xf32>, %inArg1:memref<1024xf32>)  {

    //Arrays
    %array0 = rise.in %inArg0 : !rise.array<1024, scalar<f32>>
    %array1 = rise.in %inArg1 : !rise.array<1024, scalar<f32>>

    //Zipping
    %zipFun = rise.zip #rise.nat<1024> #rise.scalar<f32> #rise.scalar<f32>
    %zippedArrays = rise.apply %zipFun, %array0, %array1

    //Reduction
    %reductionLambda = rise.lambda (%tuple, %acc) : !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> {

        %fstFun = rise.fst #rise.scalar<f32> #rise.scalar<f32>
        %sndFun = rise.snd #rise.scalar<f32> #rise.scalar<f32>

        %fst = rise.apply %fstFun, %tuple
        %snd = rise.apply %sndFun, %tuple

        %fst_unwrapped = rise.unwrap %fst
        %snd_unwrapped = rise.unwrap %snd
        %acc_unwrapped = rise.unwrap %acc

        %product = mulf %fst_unwrapped, %snd_unwrapped :f32
        %result = addf %product, %acc_unwrapped : f32
        %result_wrapped = rise.wrap %result

        rise.return %result_wrapped : !rise.scalar<f32>
    }

    %initializer = rise.literal #rise.lit<0.0>
    %reduceFun = rise.reduceSeq #rise.nat<1024> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
    %result = rise.apply %reduceFun, %reductionLambda, %initializer, %zippedArrays

    return
}

func @fused_dot() {
    //prepare output Array
    %outputArray = alloc() : memref<1xf32>
    %cst_0 = constant 0.0 : f32
    linalg.fill(%outputArray, %cst_0) : memref<1xf32>, f32

    %inArg0 = alloc() : memref<1024xf32>
    %cst_5 = constant 5.0 : f32
    linalg.fill(%inArg0, %cst_5) : memref<1024xf32>, f32

    %inArg1 = alloc() : memref<1024xf32>
    %cst_10 = constant 5.0 : f32
    linalg.fill(%inArg1, %cst_10) : memref<1024xf32>, f32

    call @rise_fun(%outputArray, %inArg0, %inArg1) : (memref<1xf32>, memref<1024xf32>, memref<1024xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<1xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// SIMPLE_DOT: Unranked Memref rank = 1 descriptor@ = {{.*}}
// SIMPLE_DOT: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// SIMPLE_DOT: [25600]

