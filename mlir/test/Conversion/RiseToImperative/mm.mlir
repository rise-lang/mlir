// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e mm -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=MM

func @print_memref_f32(memref<*xf32>)
func @rise_fun(memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
func @mm() {

    rise.fun "rise_fun" (%outArg:memref<4x4xf32>, %inA:memref<4x4xf32>, %inB:memref<4x4xf32>) {
        //Arrays
//        %A = rise.in %inA : !rise.data<array<4, array<4, float>>>
//        %B = rise.in %inB : !rise.data<array<4, array<4, float>>>

        %A = rise.literal #rise.lit<array<4.4, !rise.float, [[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]]>>
        %B = rise.literal #rise.lit<array<4.4, !rise.float, [[5,5,5,5], [5,5,5,5], [5,5,5,5], [5,5,5,5]]>>

        %m1fun = rise.lambda (%arow) : !rise.fun<data<array<4, float>> -> data<array<4, float>>> {
            %m2fun = rise.lambda (%bcol) : !rise.fun<data<array<4, float>> -> data<array<4, float>>> {

                //Zipping
                %zipFun = rise.zip #rise.nat<4> #rise.float #rise.float
                %zippedArrays = rise.apply %zipFun, %arow, %bcol

                //Multiply
                %tupleMulFun = rise.lambda (%floatTuple) : !rise.fun<data<tuple<float, float>> -> data<float>> {
                    %fstFun = rise.fst #rise.float #rise.float
                    %sndFun = rise.snd #rise.float #rise.float

                    %fst = rise.apply %fstFun, %floatTuple
                    %snd = rise.apply %sndFun, %floatTuple

                    %mulFun = rise.mul #rise.float
                    %result = rise.apply %mulFun, %snd, %fst

                    rise.return %result : !rise.data<float>
                }
                %map10TuplesToInts = rise.mapPar #rise.nat<4> #rise.tuple<float, float> #rise.float
                %multipliedArray = rise.apply %map10TuplesToInts, %tupleMulFun, %zippedArrays

                //Reduction
                %reductionAdd = rise.lambda (%summand0, %summand1) : !rise.fun<data<float> -> fun<data<float> -> data<float>>> {
                    %addFun = rise.add #rise.float
                    %doubled = rise.apply %addFun, %summand0, %summand1
                    rise.return %doubled : !rise.data<float>
                }
                %initializer = rise.literal #rise.lit<float<0>>
                %reduce10Ints = rise.reduceSeq #rise.nat<4> #rise.float #rise.float
                %result = rise.apply %reduce10Ints, %reductionAdd, %initializer, %multipliedArray

                rise.return %result : !rise.data<float>
            }
            %m2 = rise.mapPar #rise.nat<4> #rise.array<4, float> #rise.array<4, float>
            %result = rise.apply %m2, %m2fun, %B
            rise.return %result : !rise.data<array<4, array<4, float>>>
        }
        %m1 = rise.mapPar #rise.nat<4> #rise.array<4, !rise.float> #rise.array<4, !rise.float>
        %result = rise.apply %m1, %m1fun, %A
    }
    //prepare output Array
    %outputArray = alloc() : memref<4x4xf32>

    %A = alloc() : memref<4x4xf32>
    %cst_0 = constant 5.000000e+00 : f32
    linalg.fill(%A, %cst_0) : memref<4x4xf32>, f32

    %B = alloc() : memref<4x4xf32>
    %cst_1 = constant 5.000000e+00 : f32
    linalg.fill(%B, %cst_1) : memref<4x4xf32>, f32

    call @rise_fun(%outputArray, %A, %B) : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<4x4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// MM: Unranked Memref rank = 2 descriptor@ = {{.*}}
// MM: Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
// MM: {{[[100,   100,   100,   100],}}
// MM: [100,   100,   100,   100],
// MM: [100,   100,   100,   100],
// MM: [100,   100,   100,   100]]