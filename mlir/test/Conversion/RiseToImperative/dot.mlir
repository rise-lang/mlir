// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e simple_dot -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_DOT

func @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<f32>, %inArg0:memref<4xf32>, %inArg1:memref<4xf32>)  {
    rise.lowering_unit {
        //Arrays
        %array0 = rise.in %inArg0 : !rise.array<4, scalar<f32>>
        %array1 = rise.in %inArg1 : !rise.array<4, scalar<f32>>

        //Zipping
        %zipFun = rise.zip #rise.nat<4> #rise.scalar<f32> #rise.scalar<f32>
        %zippedArrays = rise.apply %zipFun, %array0, %array1

        //Multiply
        %tupleMulFun = rise.lambda (%floatTuple : !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32> {
            %fstFun = rise.fst #rise.scalar<f32> #rise.scalar<f32>
            %sndFun = rise.snd #rise.scalar<f32> #rise.scalar<f32>

            %fst = rise.apply %fstFun, %floatTuple
            %snd = rise.apply %sndFun, %floatTuple
            %result = rise.embed(%fst, %snd) {
                %result = mulf %fst, %snd : f32
                rise.return %result : f32
            } : !rise.scalar<f32>
            rise.return %result : !rise.scalar<f32>
        }

        %map10TuplesToInts = rise.mapSeq #rise.nat<4> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
        %multipliedArray = rise.apply %map10TuplesToInts, %tupleMulFun, %zippedArrays

        //Reduction
        %reductionAdd = rise.lambda (%summand0 : !rise.scalar<f32>, %summand1 : !rise.scalar<f32>) -> !rise.scalar<f32> {
            %result = rise.embed(%summand0, %summand1) {
                   %result = addf %summand0, %summand1 : f32
                   rise.return %result : f32
            } : !rise.scalar<f32>
            rise.return %result : !rise.scalar<f32>
        }
        %initializer = rise.literal #rise.lit<0.0>
        %reduce10Ints = rise.reduceSeq #rise.nat<4> #rise.scalar<f32> #rise.scalar<f32>
        %result = rise.apply %reduce10Ints, %reductionAdd, %initializer, %multipliedArray
        rise.out %outArg <- %result
        rise.return
    }
    return
}

func @simple_dot() {
    //prepare output Array
    %outputArray = alloc() : memref<f32>
    %cst_0 = constant 0.0 : f32
    linalg.fill(%outputArray, %cst_0) : memref<f32>, f32

    %inArg0 = alloc() : memref<4xf32>
    %cst_5 = constant 5.0 : f32
    linalg.fill(%inArg0, %cst_5) : memref<4xf32>, f32

    %inArg1 = alloc() : memref<4xf32>
    %cst_10 = constant 5.0 : f32
    linalg.fill(%inArg1, %cst_10) : memref<4xf32>, f32

    call @rise_fun(%outputArray, %inArg0, %inArg1) : (memref<f32>, memref<4xf32>, memref<4xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<f32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// SIMPLE_DOT: Unranked Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
// SIMPLE_DOT: [100]

