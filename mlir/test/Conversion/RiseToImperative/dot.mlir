// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e simple_dot -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_DOT

func @print_memref_f32(memref<*xf32>)
func @rise_fun(memref<1xf32>)
func @simple_dot() {

    rise.fun "rise_fun" (%outArg:memref<1xf32>) {

        //Arrays
        %array0 = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>
        %array1 = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>

        //Zipping
        %zipFun = rise.zip #rise.nat<4> #rise.float #rise.float
        %zippedArrays = rise.apply %zipFun, %array0, %array1

        //Multiply

        %tupleMulFun = rise.lambda (%floatTuple) : !rise.fun<data<tuple<float, float>> -> data<float>> {
            %fstFun = rise.fst #rise.float #rise.float
               %sndFun = rise.snd #rise.float #rise.float

               %fst = rise.apply %fstFun, %floatTuple
              %snd = rise.apply %sndFun, %floatTuple

              %mulFun = rise.mult #rise.float
              %result = rise.apply %mulFun, %snd, %fst

             rise.return %result : !rise.data<float>
            }

        %map10TuplesToInts = rise.map #rise.nat<4> #rise.tuple<float, float> #rise.float
        %multipliedArray = rise.apply %map10TuplesToInts, %tupleMulFun, %zippedArrays

        //Reduction
        %reductionAdd = rise.lambda (%summand0, %summand1) : !rise.fun<data<float> -> fun<data<float> -> data<float>>> {
            %addFun = rise.add #rise.float
            %doubled = rise.apply %addFun, %summand0, %summand1
            rise.return %doubled : !rise.data<float>
        }
        %initializer = rise.literal #rise.lit<float<0>>
        %reduce10Ints = rise.reduce #rise.nat<4> #rise.float #rise.float
        %result = rise.apply %reduce10Ints, %reductionAdd, %initializer, %multipliedArray

        rise.return %result : !rise.data<float>
    }

    //prepare output Array
    %outputArray = alloc() : memref<1xf32>
    call @rise_fun(%outputArray) : (memref<1xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<1xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// SIMPLE_DOT: Unranked Memref rank = 1 descriptor@ = {{.*}}
// SIMPLE_DOT: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// SIMPLE_DOT: [100]

