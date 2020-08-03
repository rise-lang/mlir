// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e simple_reduction -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=SIMPLE_1D_REDUCTION
func @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<f32>, %inArg:memref<1024xf32>) {
    rise.lowering_unit {
        %array0 = rise.in %inArg : !rise.array<1024, scalar<f32>>

        %reductionAdd = rise.lambda (%summand0 : !rise.scalar<f32>, %summand1 : !rise.scalar<f32>) -> !rise.scalar<f32> {
            %result = rise.embed(%summand0, %summand1) {
                %result = addf %summand0, %summand1 : f32
                rise.return %result : f32
            } : !rise.scalar<f32>
            rise.return %result : !rise.scalar<f32>
        }
        %initializer = rise.literal #rise.lit<0.0>
        %reduce4Ints = rise.reduceSeq #rise.nat<1024> #rise.scalar<f32> #rise.scalar<f32>
        %result = rise.apply %reduce4Ints, %reductionAdd, %initializer, %array0
        rise.out %outArg <- %result
        rise.return
    }
    return
}

func @simple_reduction() {

    //prepare output Array
    %outputArray = alloc() : memref<f32>
    %cst_0 = constant 0.0 : f32
    linalg.fill(%outputArray, %cst_0) : memref<f32>, f32

    %inArg0 = alloc() : memref<1024xf32>
    %cst_5 = constant 5.0 : f32
    linalg.fill(%inArg0, %cst_5) : memref<1024xf32>, f32

    call @rise_fun(%outputArray, %inArg0) : (memref<f32>, memref<1024xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<f32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// SIMPLE_1D_REDUCTION: Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
// SIMPLE_1D_REDUCTION: [5120]
