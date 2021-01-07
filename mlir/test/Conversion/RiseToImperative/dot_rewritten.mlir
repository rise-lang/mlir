// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e simple_dot -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=REWRITTEN_DOT

module {
  func private @print_memref_f32(memref<*xf32>)
  func @rise_fun(%arg0: memref<f32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>) {
    "rise.lowering_unit"() ( {
      %0 = "rise.in"(%arg1) : (memref<4xf32>) -> !rise.array<4, scalar<f32>>
      %1 = "rise.in"(%arg2) : (memref<4xf32>) -> !rise.array<4, scalar<f32>>
      %2 = "rise.zip"() {n = #rise.nat<4>, s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<array<4, scalar<f32>> -> fun<array<4, scalar<f32>> -> array<4, tuple<scalar<f32>, scalar<f32>>>>>
      %3 = "rise.apply"(%2, %0, %1) : (!rise.fun<array<4, scalar<f32>> -> fun<array<4, scalar<f32>> -> array<4, tuple<scalar<f32>, scalar<f32>>>>>, !rise.array<4, scalar<f32>>, !rise.array<4, scalar<f32>>) -> !rise.array<4, tuple<scalar<f32>, scalar<f32>>>
      %4 = "rise.literal"() {literal = #rise.lit<0.000000, scalar<f32>>} : () -> !rise.scalar<f32>
      %5 = "rise.lambda"() ( {
      ^bb0(%arg3: !rise.tuple<scalar<f32>, scalar<f32>>, %arg4: !rise.scalar<f32>):  // no predecessors
        %8 = "rise.fst"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
        %9 = "rise.snd"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
        %10 = "rise.apply"(%8, %arg3) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
        %11 = "rise.apply"(%9, %arg3) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
        %12 = "rise.embed"(%10, %11) ( {
        ^bb0(%arg5: f32, %arg6: f32):  // no predecessors
          %14 = mulf %arg5, %arg6 : f32
          "rise.return"(%14) : (f32) -> ()
        }) : (!rise.scalar<f32>, !rise.scalar<f32>) -> !rise.scalar<f32>
        %13 = "rise.embed"(%12, %arg4) ( {
        ^bb0(%arg5: f32, %arg6: f32):  // no predecessors
          %14 = addf %arg5, %arg6 : f32
          "rise.return"(%14) : (f32) -> ()
        }) : (!rise.scalar<f32>, !rise.scalar<f32>) -> !rise.scalar<f32>
        "rise.return"(%13) : (!rise.scalar<f32>) -> ()
      }) : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>
      %6 = "rise.reduceSeq"() {n = #rise.nat<4>, s = #rise.tuple<scalar<f32>, scalar<f32>>, t = #rise.scalar<f32>, to = "affine"} : () -> !rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<4, tuple<scalar<f32>, scalar<f32>>> -> scalar<f32>>>>
      %7 = "rise.apply"(%6, %5, %4, %3) : (!rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<4, tuple<scalar<f32>, scalar<f32>>> -> scalar<f32>>>>, !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>, !rise.scalar<f32>, !rise.array<4, tuple<scalar<f32>, scalar<f32>>>) -> !rise.scalar<f32>
      "rise.out"(%arg0, %7) : (memref<f32>, !rise.scalar<f32>) -> ()
      "rise.return"() : () -> ()
    }) : () -> ()
    return
  }
  func @simple_dot() {
    %0 = alloc() : memref<f32>
    %cst = constant 0.000000e+00 : f32
    linalg.fill(%0, %cst) : memref<f32>, f32
    %1 = alloc() : memref<4xf32>
    %cst_0 = constant 5.000000e+00 : f32
    linalg.fill(%1, %cst_0) : memref<4xf32>, f32
    %2 = alloc() : memref<4xf32>
    %cst_1 = constant 5.000000e+00 : f32
    linalg.fill(%2, %cst_1) : memref<4xf32>, f32
    call @rise_fun(%0, %1, %2) : (memref<f32>, memref<4xf32>, memref<4xf32>) -> ()
    %3 = memref_cast %0 : memref<f32> to memref<*xf32>
    call @print_memref_f32(%3) : (memref<*xf32>) -> ()
    return
  }
}
// REWRITTEN_DOT: Unranked Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
// REWRITTEN_DOT: [100]

