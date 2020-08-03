// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e array_times_2 -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=ARRAY_TIMES_2

func @print_memref_f32(memref<*xf32>)

func @rise_fun(%outArg: memref<6xf32>, %in: memref<6xf32>) {
    rise.lowering_unit {
        %array = rise.in %in : !rise.array<6, scalar<f32>>

        %split = rise.split #rise.nat<2> #rise.nat<3> #rise.scalar<f32>
        %array2D = rise.apply %split, %array

        %mapInnerLambda = rise.lambda (%arraySlice : !rise.array<2, scalar<f32>>) -> !rise.array<2, scalar<f32>> {
           %doubleFun = rise.lambda (%summand : !rise.scalar<f32>) -> !rise.scalar<f32> {
                %result = rise.embed(%summand) {
                       %result = addf %summand, %summand : f32
                       rise.return %result : f32
                } : !rise.scalar<f32>
               rise.return %result : !rise.scalar<f32>
           }
           %map2 = rise.mapSeq {to = "scf"} #rise.nat<2> #rise.scalar<f32> #rise.scalar<f32>
           %res = rise.apply %map2, %doubleFun, %arraySlice
           rise.return %res : !rise.array<2, scalar<f32>>
        }
        %map1 = rise.mapSeq {to = "scf"} #rise.nat<3> #rise.array<2, scalar<f32>> #rise.array<2, scalar<f32>>
        %res = rise.apply %map1, %mapInnerLambda, %array2D

        %join = rise.join #rise.nat<3> #rise.nat<2> #rise.scalar<f32>
        %flattened = rise.apply %join, %res
        rise.out %outArg <- %flattened
        rise.return
    }
    return
}

func @array_times_2() {
    //prepare output Array
    %outputArray = alloc() : memref<6xf32>
    %inputArray = alloc() : memref<6xf32>

    %memrefcst1 = alloc() : memref<f32>
    %cst1 = constant 1.000000e+00 : f32
    store %cst1, %memrefcst1[] : memref<f32>

    %cst0 = constant 0.000000e+00 : f32
    %val = alloc() : memref<f32>
    store %cst0, %val[] : memref<f32>

    %c0 = constant 0 : index
    %c16 = constant 6 : index
    %c1 = constant 1 : index
    scf.for %arg0 = %c0 to %c16 step %c1 {
        %val_loaded = load %val[] : memref<f32>
        %cst1_loaded = load %memrefcst1[] : memref<f32>
        %interm = addf %val_loaded, %cst1_loaded : f32
        store %interm, %val[] : memref<f32>
        store %interm, %inputArray[%arg0] : memref<6xf32>
    }

    call @rise_fun(%outputArray, %inputArray) : (memref<6xf32>, memref<6xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<6xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// ARRAY_TIMES_2: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [6] strides = [1] data =
// ARRAY_TIMES_2: [2, 4, 6, 8, 10, 12]

