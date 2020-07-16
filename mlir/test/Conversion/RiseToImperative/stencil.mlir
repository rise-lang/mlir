// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e stencil -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=STENCIL

func @print_memref_f32(memref<*xf32>)

func @rise_fun(%outArg: memref<11xf32>, %in: memref<9xf32>) {
    %array = rise.in %in : !rise.array<9, scalar<f32>>
    %pad = rise.pad #rise.nat<9> #rise.nat<1> #rise.nat<3> #rise.scalar<f32>
    %padded = rise.apply %pad, %array
    %slide = rise.slide #rise.nat<11> #rise.nat<3> #rise.nat<1> #rise.scalar<f32> // what does n here mean?
    %slided = rise.apply %slide, %padded

    %reduceWindow = rise.lambda (%window : !rise.array<3, scalar<f32>>) -> !rise.scalar<f32> {
        %reductionAdd = rise.lambda (%summand0 : !rise.scalar<f32>, %summand1 : !rise.scalar<f32>) -> !rise.scalar<f32> {
            %result = rise.embed(%summand0, %summand1) {
                %result = addf %summand0, %summand1 : f32
                rise.return %result : f32
            } : !rise.scalar<f32>
            rise.return %result : !rise.scalar<f32>
        }
        %initializer = rise.literal #rise.lit<0.0>
        %reduce = rise.reduceSeq #rise.nat<3> #rise.scalar<f32> #rise.scalar<f32>
        %result = rise.apply %reduce, %reductionAdd, %initializer, %window
        rise.return %result : !rise.scalar<f32>
    }
    %map = rise.mapSeq {to = "affine"}  #rise.nat<11> #rise.array<3, scalar<f32>> #rise.scalar<f32>
    %result = rise.apply %map, %reduceWindow, %slided

    rise.out %outArg <- %result
    return
}

func @stencil() {
    //prepare output Array
    %outputArray = alloc() : memref<11xf32>
    %inputArray = alloc() : memref<9xf32>

    %memrefcst1 = alloc() : memref<f32>
    %cst1 = constant 1.000000e+00 : f32
    store %cst1, %memrefcst1[] : memref<f32>

    %cst0 = constant 0.000000e+00 : f32
    %val = alloc() : memref<f32>
    store %cst0, %val[] : memref<f32>

    %c0 = constant 0 : index
    %c16 = constant 9 : index
    %c1 = constant 1 : index
    scf.for %arg0 = %c0 to %c16 step %c1 {
        %val_loaded = load %val[] : memref<f32>
        %cst1_loaded = load %memrefcst1[] : memref<f32>
        %interm = addf %val_loaded, %cst1_loaded : f32
        store %interm, %val[] : memref<f32>
        store %interm, %inputArray[%arg0] : memref<9xf32>
    }

    call @rise_fun(%outputArray, %inputArray) : (memref<11xf32>, memref<9xf32>) -> ()

    %print_me_in = memref_cast %inputArray : memref<9xf32> to memref<*xf32>
    %print_me_out = memref_cast %outputArray : memref<11xf32> to memref<*xf32>
    call @print_memref_f32(%print_me_in): (memref<*xf32>) -> ()
    call @print_memref_f32(%print_me_out): (memref<*xf32>) -> ()
    return
}
// STENCIL: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [11] strides = [1] data =
// STENCIL: [4, 6, 9, 12, 15, 18, 21, 24, 26, 27, 27]

