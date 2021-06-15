// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e mm -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=MM

func private @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<1024x1024xf32>, %inA:memref<1024x1024xf32>, %inB:memref<1024x1024xf32>) {
    rise.lowering_unit {
        %A = rise.in %inA : !rise.array<1024, array<1024, scalar<f32>>>
        %B = rise.in %inB : !rise.array<1024, array<1024, scalar<f32>>>
        %transpose = rise.transpose #rise.nat<1024> #rise.nat<1024> #rise.scalar<f32>
        %B_trans = rise.apply %transpose, %B

        %f1 = rise.lambda (%arow : !rise.array<1024, scalar<f32>>) -> !rise.array<1024, scalar<f32>> {
            %f2 = rise.lambda (%bcol : !rise.array<1024, scalar<f32>>) -> !rise.scalar<f32> {

                //Zipping
                %zipFun = rise.zip #rise.nat<1024> #rise.scalar<f32> #rise.scalar<f32>
                %zippedArrays = rise.apply %zipFun, %arow, %bcol

                //Multiply
                %f = rise.lambda (%floatTuple : !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32> {
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
                %map = rise.mapSeq #rise.nat<1024> #rise.tuple<scalar<f32>, scalar<f32>> #rise.scalar<f32>
                %multipliedArray = rise.apply %map, %f, %zippedArrays

                //Reduction
                %reductionAdd = rise.lambda (%summand0 : !rise.scalar<f32>, %summand1 : !rise.scalar<f32>) -> !rise.scalar<f32> {
                    %result = rise.embed(%summand0, %summand1) {
                           %result = addf %summand0, %summand1 : f32
                           rise.return %result : f32
                    } : !rise.scalar<f32>
                    rise.return %result : !rise.scalar<f32>
                }
                %initializer = rise.literal #rise.lit<0.0>
                %reduce10Ints = rise.reduceSeq #rise.nat<1024> #rise.scalar<f32> #rise.scalar<f32>
                %result = rise.apply %reduce10Ints, %reductionAdd, %initializer, %multipliedArray

                rise.return %result : !rise.scalar<f32>
            }
            %m2 = rise.mapSeq #rise.nat<1024> #rise.array<1024, scalar<f32>> #rise.scalar<f32>
            %result = rise.apply %m2, %f2, %B_trans
            rise.return %result : !rise.array<1024, scalar<f32>>
        }
        %m1 = rise.mapSeq #rise.nat<1024> #rise.array<1024, scalar<f32>> #rise.array<1024, scalar<f32>>
        %result = rise.apply %m1, %f1, %A
        rise.out %outArg <- %result
        rise.return
    }
    return
}
func @mm() {
    //prepare output Array
    %outputArray = memref.alloc() : memref<1024x1024xf32>

    %A = memref.alloc() : memref<1024x1024xf32>
    %cst_0 = constant 5.000000e+00 : f32
    linalg.fill(%A, %cst_0) : memref<1024x1024xf32>, f32

    %B = memref.alloc() : memref<1024x1024xf32>
    %cst_1 = constant 5.000000e+00 : f32
    linalg.fill(%B, %cst_1) : memref<1024x1024xf32>, f32

    call @rise_fun(%outputArray, %A, %B) : (memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>) -> ()

    %print_me = memref.cast %outputArray : memref<1024x1024xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// MM: Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [1024, 1024] strides = [1024, 1] data =
// MM: {{[[100,   100,   100,   100],}}
// MM: [100,   100,   100,   100],
// MM: [100,   100,   100,   100],
// MM: [100,   100,   100,   100]]