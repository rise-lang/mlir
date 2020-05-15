// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e mm -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=MM

func @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<2048x2048xf32>, %inA:memref<2048x2048xf32>, %inB:memref<2048x2048xf32>) {
    %A = rise.in %inA : !rise.array<2048, array<2048, scalar<f32>>>
    %B = rise.in %inB : !rise.array<2048, array<2048, scalar<f32>>>

    %f1 = rise.lambda (%arow) : !rise.fun<array<2048, scalar<f32>> -> array<2048, scalar<f32>>> {
        %f2 = rise.lambda (%bcol) : !rise.fun<array<2048, scalar<f32>> -> array<2048, scalar<f32>>> {

            //Zipping
            %zipFun = rise.zip #rise.nat<2048> #rise.scalar<f32> #rise.scalar<f32>
            %zippedArrays = rise.apply %zipFun, %arow, %bcol

            //Multiply
            %f = rise.lambda (%floatTuple) : !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>> {
                %fstFun = rise.fst #rise.scalar<f32> #rise.scalar<f32>
                %sndFun = rise.snd #rise.scalar<f32> #rise.scalar<f32>

                %fst = rise.apply %fstFun, %floatTuple
                %snd = rise.apply %sndFun, %floatTuple

                %fst_unwrapped = rise.unwrap %fst
                %snd_unwrapped = rise.unwrap %snd
                %result = mulf %fst_unwrapped, %snd_unwrapped :f32
                %result_wrapped = rise.wrap %result

                rise.return %result_wrapped : !rise.scalar<f32>
            }
            %map = rise.mapPar #rise.nat<2048> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
            %multipliedArray = rise.apply %map, %f, %zippedArrays

            //Reduction
            %reductionAdd = rise.lambda (%summand0, %summand1) : !rise.fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>> {
                %summand0_unwrapped = rise.unwrap %summand0
                %summand1_unwrapped = rise.unwrap %summand1
                %result = addf %summand0_unwrapped, %summand1_unwrapped : f32
                %result_wrapped = rise.wrap %result
                rise.return %result_wrapped : !rise.scalar<f32>
            }
            %initializer = rise.literal #rise.lit<0.0>
            %reduce10Ints = rise.reduceSeq #rise.nat<2048> #rise.scalar<f32> #rise.scalar<f32>
            %result = rise.apply %reduce10Ints, %reductionAdd, %initializer, %multipliedArray

            rise.return %result : !rise.scalar<f32>
        }
        %m2 = rise.mapPar #rise.nat<2048> #rise.array<2048, scalar<f32>> #rise.array<2048, scalar<f32>>
        %result = rise.apply %m2, %f2, %B
        rise.return %result : !rise.array<2048, array<2048, scalar<f32>>>
    }
    %m1 = rise.mapPar #rise.nat<2048> #rise.array<2048, scalar<f32>> #rise.array<2048, scalar<f32>>
    %result = rise.apply %m1, %f1, %A
    return
}
func @mm() {
    //prepare output Array
    %outputArray = alloc() : memref<2048x2048xf32>

    %A = alloc() : memref<2048x2048xf32>
    %cst_0 = constant 5.000000e+00 : f32
    linalg.fill(%A, %cst_0) : memref<2048x2048xf32>, f32

    %B = alloc() : memref<2048x2048xf32>
    %cst_1 = constant 5.000000e+00 : f32
    linalg.fill(%B, %cst_1) : memref<2048x2048xf32>, f32

    call @rise_fun(%outputArray, %A, %B) : (memref<2048x2048xf32>, memref<2048x2048xf32>, memref<2048x2048xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<2048x2048xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// MM: Unranked Memref rank = 2 descriptor@ = {{.*}}
// MM: Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
// MM: {{[[100,   100,   100,   100],}}
// MM: [100,   100,   100,   100],
// MM: [100,   100,   100,   100],
// MM: [100,   100,   100,   100]]