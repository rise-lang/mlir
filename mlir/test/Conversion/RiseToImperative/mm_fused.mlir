// RUN: mlir-opt %s
// mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e mm -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext,/opt/intel/lib/intel64_lin/libmkl_intel_ilp64.so,/home/martin/development/phd/projects/MLIR/performance_measuring/dylib/measure_lib.so | FileCheck %s --check-prefix=MM_irreg


func @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<2048x2048xf32>, %inA:memref<2048x2048xf32>, %inB:memref<2048x2048xf32>) {
    %outputArray1 = alloc() : memref<2048x2048xf32>
    %outputArray = alloc() : memref<2048x2048xf32>
    %outputArray2 = alloc() : memref<2048x2048xf32>

    %A = rise.in %inA : !rise.array<2048, array<2048, scalar<f32>>>
    %B = rise.in %inB : !rise.array<2048, array<2048, scalar<f32>>>
    %transpose = rise.transpose #rise.nat<2048> #rise.nat<2048> #rise.scalar<f32>
    %B_trans = rise.apply %transpose, %B


    %m1fun = rise.lambda (%arow : !rise.array<2048, scalar<f32>>) -> !rise.array<2048, scalar<f32>> {
        %m2fun = rise.lambda (%bcol : !rise.array<2048, scalar<f32>>) -> !rise.array<2048, scalar<f32>> {

            //Zipping
            %zipFun = rise.zip #rise.nat<2048> #rise.scalar<f32> #rise.scalar<f32>
            %zippedArrays = rise.apply %zipFun, %arow, %bcol

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
            %reduceFun = rise.reduceSeq {to = "affine"}  #rise.nat<2048> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
            %result = rise.apply %reduceFun, %reductionLambda, %initializer, %zippedArrays

            rise.return %result : !rise.scalar<f32>
        }
        %m2 = rise.mapSeq {to = "affine"}  #rise.nat<2048> #rise.array<2048, scalar<f32>> #rise.array<2048, scalar<f32>>
        %result = rise.apply %m2, %m2fun, %B_trans
        rise.return %result : !rise.array<2048, array<2048, scalar<f32>>>
    }
    %m1 = rise.mapSeq {to = "affine"}  #rise.nat<2048> #rise.array<2048, scalar<f32>> #rise.array<2048, scalar<f32>>
    %result = rise.apply %m1, %m1fun, %A
    rise.out %outArg <- %result
    return
}
func @rtclock() -> (f64)
func @print_flops(f64,f64,i64)
func @mm() {
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
    scf.for %arg0 = %c0 to %c16 step %c1 {
        scf.for %arg1 = %c0 to %c16 step %c1 {
            %val_loaded = load %val[] : memref<f32>
            %cst1_loaded = load %memrefcst1[] : memref<f32>
            %interm = addf %val_loaded, %cst1_loaded : f32
            store %interm, %val[] : memref<f32>
            store %interm, %A[%arg1, %arg0] : memref<2048x2048xf32>
        }
    }

    %B = alloc() : memref<2048x2048xf32>
    scf.for %i = %c0 to %c16 step %c1 {
        scf.for %arg1 = %c0 to %c16 step %c1 {
            %val_loaded = load %val[] : memref<f32>
            %cst1_loaded = load %memrefcst1[] : memref<f32>
            %interm = addf %val_loaded, %cst1_loaded : f32
            store %interm, %val[] : memref<f32>
            store %interm, %B[%i, %arg1] : memref<2048x2048xf32>
        }
    }

    %t0 = call @rtclock() : () -> (f64)
    call @rise_fun(%outputArray, %A, %B) : (memref<2048x2048xf32>, memref<2048x2048xf32>, memref<2048x2048xf32>) -> ()
    %t1 = call @rtclock() : () -> (f64)
    %ci1 = constant 17179869184 : i64 // Number of flops to compute

    %print_me = memref_cast %outputArray : memref<2048x2048xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()

    call @print_flops(%t0, %t1, %ci1): (f64,f64,i64) -> ()
    return
}
