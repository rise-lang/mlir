// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e mm -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=MM_irreg

func @print_memref_f32(memref<*xf32>)

// A:10x2 * B:2x5 = C:10x5
func @rise_fun(%outArg:memref<10x5xf32>, %inA:memref<10x2xf32>, %inB:memref<2x5xf32>) {
    %A = rise.in %inA : !rise.array<10, array<2, scalar<f32>>>
    %B = rise.in %inB : !rise.array<2, array<5, scalar<f32>>>
    %transpose = rise.transpose #rise.nat<2> #rise.nat<5> #rise.scalar<f32>
    %B_trans = rise.apply %transpose, %B

    %m1fun = rise.lambda (%arow : !rise.array<2, scalar<f32>>) -> !rise.array<2, scalar<f32>> {
        %m2fun = rise.lambda (%bcol : !rise.array<2, scalar<f32>>) -> !rise.array<2, scalar<f32>> {

            //Zipping
            %zipFun = rise.zip #rise.nat<2> #rise.scalar<f32> #rise.scalar<f32>
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
            %reduceFun = rise.reduceSeq {to = "affine"}  #rise.nat<2> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
            %result = rise.apply %reduceFun, %reductionLambda, %initializer, %zippedArrays

            rise.return %result : !rise.scalar<f32>
        }
        %m2 = rise.mapSeq {to = "affine"}  #rise.nat<5> #rise.array<2, scalar<f32>> #rise.array<2, scalar<f32>>
        %result = rise.apply %m2, %m2fun, %B_trans
        rise.return %result : !rise.array<5, array<2, scalar<f32>>>
    }
    %m1 = rise.mapSeq {to = "affine"}  #rise.nat<10> #rise.array<2, scalar<f32>> #rise.array<2, scalar<f32>>
    %result = rise.apply %m1, %m1fun, %A
    rise.out %outArg <- %result
    return
}
func @rtclock() -> (f64)
func @print_flops(f64,f64,i64)
func @mm() {
    //prepare output Array
    %outputArray = alloc() : memref<10x5xf32>

    %A = alloc() : memref<10x2xf32>

    %cst0 = constant 0.000000e+00 : f32
    %memrefcst1 = alloc() : memref<f32>
    %cst1 = constant 1.000000e+00 : f32
    store %cst1, %memrefcst1[] : memref<f32>

    %val = alloc() : memref<f32>
    store %cst0, %val[] : memref<f32>

    %c0 = constant 0 : index
    %cArows = constant 10 : index
    %cAcols = constant 2 : index
    %c1 = constant 1 : index
    scf.for %arg0 = %c0 to %cArows step %c1 {
        scf.for %arg1 = %c0 to %cAcols step %c1 {
            %val_loaded = load %val[] : memref<f32>
            %cst1_loaded = load %memrefcst1[] : memref<f32>
            %interm = addf %val_loaded, %cst1_loaded : f32
            store %interm, %val[] : memref<f32>
            store %interm, %A[%arg0, %arg1] : memref<10x2xf32>
        }
    }

    %cBrows = constant 2 : index
    %cBcols = constant 5 : index
    %B = alloc() : memref<2x5xf32>

    scf.for %arg0 = %c0 to %cBrows step %c1 {
        scf.for %arg1 = %c0 to %cBcols step %c1 {
            %val_loaded = load %val[] : memref<f32>
            %cst1_loaded = load %memrefcst1[] : memref<f32>
            %interm = addf %val_loaded, %cst1_loaded : f32
            store %interm, %val[] : memref<f32>
            store %interm, %B[%arg0, %arg1] : memref<2x5xf32>
        }
    }

    call @rise_fun(%outputArray, %A, %B) : (memref<10x5xf32>, memref<10x2xf32>, memref<2x5xf32>) -> ()

    %print_meA = memref_cast %A : memref<10x2xf32> to memref<*xf32>
    call @print_memref_f32(%print_meA): (memref<*xf32>) -> ()

    %print_meB = memref_cast %B : memref<2x5xf32> to memref<*xf32>
    call @print_memref_f32(%print_meB): (memref<*xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<10x5xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()

    return
}

// In the test we multiply A:
//Unranked Memref base@ = 0x55be46ad9f30 rank = 2 offset = 0 sizes = [10, 2] strides = [2, 1] data =
//[[1,   2],
// [3,   4],
// [5,   6],
// [7,   8],
// [9,   10],
// [11,   12],
// [13,   14],
// [15,   16],
// [17,   18],
// [19,   20]]

// with B:
//Unranked Memref base@ = 0x55be46a13740 rank = 2 offset = 0 sizes = [2, 5] strides = [5, 1] data =
//[[21,   22,   23,   24,   25],
// [26,   27,   28,   29,   30]]

// equals C:
// MM_irreg: Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [10, 5] strides = [5, 1] data =
// MM_irreg: {{[[73,   76,   79,   82,   85],}}
// MM_irreg: [167,   174,   181,   188,   195],
// MM_irreg: [261,   272,   283,   294,   305],
// MM_irreg: [355,   370,   385,   400,   415],
// MM_irreg: [449,   468,   487,   506,   525],
// MM_irreg: [543,   566,   589,   612,   635],
// MM_irreg: [637,   664,   691,   718,   745],
// MM_irreg: [731,   762,   793,   824,   855],
// MM_irreg: [825,   860,   895,   930,   965],
// MM_irreg: [919,   958,   997,   1036,   1075]]
