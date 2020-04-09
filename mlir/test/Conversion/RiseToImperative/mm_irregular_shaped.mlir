// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e mm -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext,/home/martin/development/phd/projects/MLIR/performance_measuring/dylib/measure_lib.so

func @print_memref_f32(memref<*xf32>)
func @rise_fun(memref<4x5xf32>, memref<4x3xf32>, memref<5x3xf32>)
func @rtclock() -> (f64)
func @mm() {

    rise.fun "rise_fun" (%outArg:memref<4x5xf32>, %inA:memref<4x3xf32>, %inB:memref<5x3xf32>) {
        //Arrays
        %A = rise.in %inA : !rise.array<4, array<3, scalar<f32>>>
        %B = rise.in %inB : !rise.array<5, array<3, scalar<f32>>>

        %m1fun = rise.lambda (%arow) : !rise.fun<array<3, scalar<f32>> -> array<3, scalar<f32>>> {
            %m2fun = rise.lambda (%bcol) : !rise.fun<array<3, scalar<f32>> -> array<3, scalar<f32>>> {

                //Zipping
                %zipFun = rise.zip #rise.nat<3> #rise.scalar<f32> #rise.scalar<f32>
                %zippedArrays = rise.apply %zipFun, %arow, %bcol

                //Multiply
                %tupleMulFun = rise.lambda (%floatTuple) : !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>> {
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
                %map10TuplesToInts = rise.mapPar #rise.nat<3> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
                %multipliedArray = rise.apply %map10TuplesToInts, %tupleMulFun, %zippedArrays

                //Reduction
                %reductionAdd = rise.lambda (%summand0, %summand1) : !rise.fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>> {
                    %summand0_unwrapped = rise.unwrap %summand0
                    %summand1_unwrapped = rise.unwrap %summand1
                    %result = addf %summand0_unwrapped, %summand1_unwrapped : f32
                    %result_wrapped = rise.wrap %result
                    rise.return %result_wrapped : !rise.scalar<f32>
                }
                %initializer = rise.literal #rise.lit<0.0>
                %reduce10Ints = rise.reduceSeq #rise.nat<3> #rise.scalar<f32> #rise.scalar<f32>
                %result = rise.apply %reduce10Ints, %reductionAdd, %initializer, %multipliedArray

                rise.return %result : !rise.scalar<f32>
            }
            %m2 = rise.mapPar #rise.nat<5> #rise.array<3, scalar<f32>> #rise.array<3, scalar<f32>>
            %result = rise.apply %m2, %m2fun, %B
            rise.return %result : !rise.array<5, array<3, scalar<f32>>>
        }
        %m1 = rise.mapPar #rise.nat<4> #rise.array<3, scalar<f32>> #rise.array<3, scalar<f32>>
        %result = rise.apply %m1, %m1fun, %A
        rise.return %result : !rise.array<4, array<3, scalar<f32>>>
    }
    //prepare output Array
    %outputArray = alloc() : memref<4x5xf32>


    %A = alloc() : memref<4x3xf32>

    %cst0 = constant 0.000000e+00 : f32
    %memrefcst1 = alloc() : memref<f32>
    %cst1 = constant 1.000000e+00 : f32
    store %cst1, %memrefcst1[] : memref<f32>

    %val = alloc() : memref<f32>
    store %cst0, %val[] : memref<f32>

    %c0 = constant 0 : index
    %c3 = constant 3 : index
    %c4 = constant 4 : index
    %c5 = constant 5 : index
    %c1 = constant 1 : index
    loop.for %arg0 = %c0 to %c4 step %c1 {
        loop.for %arg1 = %c0 to %c3 step %c1 {
            %val_loaded = load %val[] : memref<f32>
            %cst1_loaded = load %memrefcst1[] : memref<f32>
            %interm = addf %val_loaded, %cst1_loaded : f32
            store %interm, %val[] : memref<f32>
            // transposed here
            store %interm, %A[%arg0, %arg1] : memref<4x3xf32>
        }
    }

//    %A = alloc() : memref<4x4xf32>
//    %cst_0 = constant 5.000000e+00 : f32
//    linalg.fill(%A, %cst_0) : memref<4x4xf32>, f32

    %B = alloc() : memref<5x3xf32>
//    %cst_1 = constant 5.000000e+00 : f32
//    linalg.fill(%B, %cst_1) : memref<4x4xf32>, f32
    loop.for %arg0 = %c0 to %c3 step %c1 {
        loop.for %arg1 = %c0 to %c5 step %c1 {
            %val_loaded = load %val[] : memref<f32>
            %cst1_loaded = load %memrefcst1[] : memref<f32>
            %interm = addf %val_loaded, %cst1_loaded : f32
            store %interm, %val[] : memref<f32>
            store %interm, %B[%arg1, %arg0] : memref<5x3xf32>
        }
    }

    %t0 = call @rtclock() : () -> (f64)
    call @rise_fun(%outputArray, %A, %B) : (memref<4x5xf32>, memref<4x3xf32>, memref<5x3xf32>) -> ()
    %t1 = call @rtclock() : () -> (f64)

    %print_me_A = memref_cast %A : memref<4x3xf32> to memref<*xf32>
    call @print_memref_f32(%print_me_A): (memref<*xf32>) -> ()

    %print_me_B = memref_cast %B : memref<5x3xf32> to memref<*xf32>
    call @print_memref_f32(%print_me_B): (memref<*xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<4x5xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
