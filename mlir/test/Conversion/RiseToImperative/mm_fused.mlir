// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e mm -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext,/home/martin/development/phd/projects/MLIR/performance_measuring/dylib/measure_lib.so

func @print_memref_f32(memref<*xf32>)
//func @rise_fun(%_outArg:memref<2048x2048xf32>, %_inA:memref<2048x2048xf32>, %_inB:memref<2048x2048xf32>) {return}
func @rise_fun(%outArg:memref<2048x2048xf32>, %inA:memref<2048x2048xf32>, %inB:memref<2048x2048xf32>) {
    %A = rise.in %inA : !rise.array<2048, array<2048, scalar<f32>>>
    %B = rise.in %inB : !rise.array<2048, array<2048, scalar<f32>>>

    %m1fun = rise.lambda (%arow) : !rise.fun<array<2048, scalar<f32>> -> array<2048, scalar<f32>>> {
        %m2fun = rise.lambda (%bcol) : !rise.fun<array<2048, scalar<f32>> -> array<2048, scalar<f32>>> {

            //Zipping
            %zipFun = rise.zip #rise.nat<2048> #rise.scalar<f32> #rise.scalar<f32>
            %zippedArrays = rise.apply %zipFun, %arow, %bcol

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
            %reduceFun = rise.reduceSeq {to = "loop"}  #rise.nat<2048> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
            %result = rise.apply %reduceFun, %reductionLambda, %initializer, %zippedArrays

            rise.return %result : !rise.scalar<f32>
        }
        %m2 = rise.mapSeq {to = "loop"}  #rise.nat<2048> #rise.array<2048, scalar<f32>> #rise.array<2048, scalar<f32>>
        %result = rise.apply %m2, %m2fun, %B
        rise.return %result : !rise.array<2048, array<2048, scalar<f32>>>
    }
    %m1 = rise.mapSeq {to = "loop"}  #rise.nat<2048> #rise.array<2048, scalar<f32>> #rise.array<2048, scalar<f32>>
    %result = rise.apply %m1, %m1fun, %A
//    rise.return %result : !rise.array<2048, array<2048, scalar<f32>>>
    return
}
func @rtclock() -> (f64)
func @print_flops(f64,f64,i64)
func @mm() {

// zip xs ys |> map (fun(p => fst(p) * snd(p))) |> reduce (fun(x => fun(acc => x + acc))) 0.0
//  vs.
// zip xs ys |> reduceSeq (fun(p => fun(acc => (fst(p) * snd(p)) + acc))) 0.0
//
//    rise.fun "rise_fun" (%outArg:memref<2048x2048xf32>, %inA:memref<2048x2048xf32>, %inB:memref<2048x2048xf32>) {
//        //Arrays
//        %A = rise.in %inA : !rise.array<2048, array<2048, scalar<f32>>>
//        %B = rise.in %inB : !rise.array<2048, array<2048, scalar<f32>>>
//
//        %m1fun = rise.lambda (%arow) : !rise.fun<array<2048, scalar<f32>> -> array<2048, scalar<f32>>> {
//            %m2fun = rise.lambda (%bcol) : !rise.fun<array<2048, scalar<f32>> -> array<2048, scalar<f32>>> {
//
//                //Zipping
//                %zipFun = rise.zip #rise.nat<2048> #rise.scalar<f32> #rise.scalar<f32>
//                %zippedArrays = rise.apply %zipFun, %arow, %bcol
//
//                //Reduction
//                %reductionLambda = rise.lambda (%tuple, %acc) : !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> {
//
//                    %fstFun = rise.fst #rise.scalar<f32> #rise.scalar<f32>
//                    %sndFun = rise.snd #rise.scalar<f32> #rise.scalar<f32>
//
//                    %fst = rise.apply %fstFun, %tuple
//                    %snd = rise.apply %sndFun, %tuple
//
//                    %fst_unwrapped = rise.unwrap %fst
//                    %snd_unwrapped = rise.unwrap %snd
//                    %acc_unwrapped = rise.unwrap %acc
//
//                    %product = mulf %fst_unwrapped, %snd_unwrapped :f32
//                    %result = addf %product, %acc_unwrapped : f32
//                    %result_wrapped = rise.wrap %result
//
//                    rise.return %result_wrapped : !rise.scalar<f32>
//                }
//
//                %initializer = rise.literal #rise.lit<0.0>
//                %reduceFun = rise.reduceSeq {to = "affine"}  #rise.nat<2048> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
//                %result = rise.apply %reduceFun, %reductionLambda, %initializer, %zippedArrays
//
//                rise.return %result : !rise.scalar<f32>
//            }
//            %m2 = rise.mapSeq {to = "affine"}  #rise.nat<2048> #rise.array<2048, scalar<f32>> #rise.array<2048, scalar<f32>>
//            %result = rise.apply %m2, %m2fun, %B
//            rise.return %result : !rise.array<2048, array<2048, scalar<f32>>>
//        }
//        %m1 = rise.mapSeq {to = "affine"}  #rise.nat<2048> #rise.array<2048, scalar<f32>> #rise.array<2048, scalar<f32>>
//        %result = rise.apply %m1, %m1fun, %A
//        rise.return %result : !rise.array<2048, array<2048, scalar<f32>>>
//    }

    //prepare output Array
    %outputArray = alloc() : memref<2048x2048xf32>


    %A = alloc() : memref<2048x2048xf32>

    %cst0 = constant 0.000000e+00 : f32
    %memrefcst1 = alloc() : memref<f32>
    %cst1 = constant 1.000000e+00 : f32
    store %cst1, %memrefcst1[] : memref<f32>

    %val = alloc() : memref<f32>
    store %cst0, %val[] : memref<f32>

    %c0 = constant 0 : index
    %c16 = constant 2048 : index
    %c1 = constant 1 : index
    loop.for %arg0 = %c0 to %c16 step %c1 {
        loop.for %arg1 = %c0 to %c16 step %c1 {
            %val_loaded = load %val[] : memref<f32>
            %cst1_loaded = load %memrefcst1[] : memref<f32>
            %interm = addf %val_loaded, %cst1_loaded : f32
            store %interm, %val[] : memref<f32>
            // transposed here
            store %interm, %A[%arg1, %arg0] : memref<2048x2048xf32>
        }
    }

//    %A = alloc() : memref<4x4xf32>
//    %cst_0 = constant 5.000000e+00 : f32
//    linalg.fill(%A, %cst_0) : memref<4x4xf32>, f32

    %B = alloc() : memref<2048x2048xf32>
//    %cst_1 = constant 5.000000e+00 : f32
//    linalg.fill(%B, %cst_1) : memref<4x4xf32>, f32
    loop.for %arg0 = %c0 to %c16 step %c1 {
        loop.for %arg1 = %c0 to %c16 step %c1 {
            %val_loaded = load %val[] : memref<f32>
            %cst1_loaded = load %memrefcst1[] : memref<f32>
            %interm = addf %val_loaded, %cst1_loaded : f32
            store %interm, %val[] : memref<f32>
            store %interm, %B[%arg0, %arg1] : memref<2048x2048xf32>
        }
    }

    %t0 = call @rtclock() : () -> (f64)
    call @rise_fun(%outputArray, %A, %B) : (memref<2048x2048xf32>, memref<2048x2048xf32>, memref<2048x2048xf32>) -> ()
    %t1 = call @rtclock() : () -> (f64)
    %ci1 = constant 17179869184 : i64 // Number of flops to compute


    %print_me = memref_cast %outputArray : memref<2048x2048xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()

    call @print_flops(%t0, %t1, %ci1): (f64,f64,i64) -> ()

//    %print_me_A = memref_cast %A : memref<4x4xf64> to memref<*xf64
//    call @print_memref_f64(%print_me_A): (memref<*xf64>) -> ()
//
//    %print_me_B = memref_cast %B : memref<4x4xf64> to memref<*xf64>
//    call @print_memref_f64(%print_me_B): (memref<*xf64>) -> ()
//

    return
}
