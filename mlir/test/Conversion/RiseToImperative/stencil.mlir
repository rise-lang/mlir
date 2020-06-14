// RUN: mlir-opt %s
//-convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e array_times_2 -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=STENCIL

func @print_memref_f32(memref<*xf32>)

func @rise_fun(%outArg: memref<9xf32>, %in: memref<9xf32>) {
    // This way we dont have to handle moving the block arguments anymore.
    %array = rise.in %in : !rise.array<9, scalar<f32>>
    %padVal = rise.literal #rise.lit<5.0>
    %pad = rise.pad #rise.nat<9> #rise.nat<1> #rise.nat<1> #rise.scalar<f32>
    %padded = rise.apply %pad, %padVal, %array
    %slide = rise.slide #rise.nat<9> #rise.nat<3> #rise.nat<1> #rise.scalar<f32> // what does n here mean?
    %slided = rise.apply %slide, %padded

    %reduceWindow = rise.lambda (%window : !rise.array<3, scalar<f32>>) -> !rise.scalar<f32> {
        %reductionAdd = rise.lambda (%summand0 : !rise.scalar<f32>, %summand1 : !rise.scalar<f32>) -> !rise.scalar<f32> {
            %result = rise.embed(%summand0, %summand1) {
                %result = addf %summand0, %summand1 : f32
                rise.return %result : f32
            }
            rise.return %result : !rise.scalar<f32>
        }
        %initializer = rise.literal #rise.lit<0.0>
        %reduce = rise.reduceSeq #rise.nat<3> #rise.scalar<f32> #rise.scalar<f32>
        %result = rise.apply %reduce, %reductionAdd, %initializer, %window
    }
    %map = rise.mapSeq {to = "affine"}  #rise.nat<9> #rise.array<3, scalar<f32>> #rise.scalar<f32>
    %result = rise.apply %map, %reduceWindow, %slided

    rise.out %outArg <- %result
    return
}

func @stencil() {
    //prepare output Array
    %outputArray = alloc() : memref<9xf32>

    %inputArray = alloc() : memref<9xf32>
    %cst = constant 5.0 : f32
    linalg.fill(%inputArray, %cst) : memref<9xf32>, f32

    call @rise_fun(%outputArray, %inputArray) : (memref<9xf32>, memref<9xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<9xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// STENCIL: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [9] strides = [1] data =
// STENCIL: [15, 15, 15, 15]

