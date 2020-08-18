// RUN: mlir-opt %s
//-convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e array_times_2 -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=ARRAY_TIMES_2

func @print_memref_f32(memref<*xf32>)

func @rise_fun(%outArg:memref<122x15xf32>, %inArg:memref<15x122xf32>) {
    rise.lowering_unit {
        %array2D = rise.in %inArg : !rise.array<15, array<122, scalar<f32>>>

        // map(map(id))
        %mapInnerLambda = rise.lambda (%arraySlice : !rise.array<122, scalar<f32>>) -> !rise.array<122, scalar<f32>> {
           %doubleFun = rise.lambda (%id : !rise.scalar<f32>) -> !rise.scalar<f32> {
                %result = rise.embed(%id) {
                       %doubled = addf %id, %id : f32
                       %result = subf %doubled, %id : f32
                       rise.return %result : f32
                } : !rise.scalar<f32>
               rise.return %result : !rise.scalar<f32>
           }
           %map2 = rise.mapSeq #rise.nat<122> #rise.scalar<f32> #rise.scalar<f32>
           %res = rise.apply %map2, %doubleFun, %arraySlice
           rise.return %res : !rise.array<122, scalar<f32>>
        }
        %map1 = rise.mapSeq #rise.nat<15> #rise.array<122, scalar<f32>> #rise.array<122, scalar<f32>>
        %id = rise.apply %map1, %mapInnerLambda, %array2D

        %transpose = rise.transpose #rise.nat<15> #rise.nat<122> #rise.scalar<f32>
        %transposed = rise.apply %transpose, %id
        rise.out %outArg <- %transposed

//        rise.out %outArg <- %id
        rise.return
    }
    return
}

func @array_times_2() {
    //prepare output Array
    %outputArray = alloc() : memref<122x15xf32>
    %inputArray = alloc() : memref<15x122xf32>


    %memrefcst1 = alloc() : memref<f32>
    %cst1 = constant 1.000000e+00 : f32
    store %cst1, %memrefcst1[] : memref<f32>

    %cst0 = constant 0.000000e+00 : f32
    %val = alloc() : memref<f32>
    store %cst0, %val[] : memref<f32>

    %c0 = constant 0 : index
    %c3 = constant 15 : index
    %c2 = constant 122 : index
    %c1 = constant 1 : index
    scf.for %arg0 = %c0 to %c3 step %c1 {
        scf.for %arg1 = %c0 to %c2 step %c1 {
            %val_loaded = load %val[] : memref<f32>
            %cst1_loaded = load %memrefcst1[] : memref<f32>
            %interm = addf %val_loaded, %cst1_loaded : f32
            store %interm, %val[] : memref<f32>
            store %interm, %inputArray[%arg0, %arg1] : memref<15x122xf32>
        }
    }

    call @rise_fun(%outputArray, %inputArray) : (memref<122x15xf32>, memref<15x122xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<122x15xf32> to memref<*xf32>
    %print_me_input = memref_cast %inputArray : memref<15x122xf32> to memref<*xf32>
    call @print_memref_f32(%print_me_input): (memref<*xf32>) -> ()
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// ARRAY_TIMES_2: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [6] strides = [1] data =
// ARRAY_TIMES_2: [2, 4, 6, 8, 10, 15]

