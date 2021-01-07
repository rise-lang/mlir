// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e mm -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=MM

module {
  func private @print_memref_f32(memref<*xf32>)
  func @rise_fun(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>) {
    "rise.lowering_unit"() ( {
      %0 = "rise.in"(%arg1) : (memref<4x4xf32>) -> !rise.array<4, array<4, scalar<f32>>>
      %1 = "rise.in"(%arg2) : (memref<4x4xf32>) -> !rise.array<4, array<4, scalar<f32>>>
      %2 = "rise.lambda"() ( {
      ^bb0(%arg3: !rise.array<4, scalar<f32>>):  // no predecessors
        %9 = "rise.lambda"() ( {
        ^bb0(%arg4: !rise.array<4, scalar<f32>>):  // no predecessors
          %12 = "rise.zip"() {n = #rise.nat<4>, s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<array<4, scalar<f32>> -> fun<array<4, scalar<f32>> -> array<4, tuple<scalar<f32>, scalar<f32>>>>>
          %13 = "rise.apply"(%12, %arg3, %arg4) : (!rise.fun<array<4, scalar<f32>> -> fun<array<4, scalar<f32>> -> array<4, tuple<scalar<f32>, scalar<f32>>>>>, !rise.array<4, scalar<f32>>, !rise.array<4, scalar<f32>>) -> !rise.array<4, tuple<scalar<f32>, scalar<f32>>>
          %14 = "rise.lambda"() ( {
          ^bb0(%arg5: !rise.tuple<scalar<f32>, scalar<f32>>):  // no predecessors
            %21 = "rise.fst"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
            %22 = "rise.snd"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
            %23 = "rise.apply"(%21, %arg5) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
            %24 = "rise.apply"(%22, %arg5) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
            %25 = "rise.embed"(%23, %24) ( {
            ^bb0(%arg6: f32, %arg7: f32):  // no predecessors
              %26 = mulf %arg6, %arg7 : f32
              "rise.return"(%26) : (f32) -> ()
            }) : (!rise.scalar<f32>, !rise.scalar<f32>) -> !rise.scalar<f32>
            "rise.return"(%25) : (!rise.scalar<f32>) -> ()
          }) : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
          %15 = "rise.mapSeq"() {n = #rise.nat<4>, s = #rise.tuple<scalar<f32>, scalar<f32>>, t = #rise.scalar<f32>} : () -> !rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>> -> fun<array<4, tuple<scalar<f32>, scalar<f32>>> -> array<4, scalar<f32>>>>
          %16 = "rise.apply"(%15, %14, %13) : (!rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>> -> fun<array<4, tuple<scalar<f32>, scalar<f32>>> -> array<4, scalar<f32>>>>, !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.array<4, tuple<scalar<f32>, scalar<f32>>>) -> !rise.array<4, scalar<f32>>
          %17 = "rise.lambda"() ( {
          ^bb0(%arg5: !rise.scalar<f32>, %arg6: !rise.scalar<f32>):  // no predecessors
            %21 = "rise.embed"(%arg5, %arg6) ( {
            ^bb0(%arg7: f32, %arg8: f32):  // no predecessors
              %22 = addf %arg7, %arg8 : f32
              "rise.return"(%22) : (f32) -> ()
            }) : (!rise.scalar<f32>, !rise.scalar<f32>) -> !rise.scalar<f32>
            "rise.return"(%21) : (!rise.scalar<f32>) -> ()
          }) : () -> !rise.fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>>
          %18 = "rise.literal"() {literal = #rise.lit<0.000000, scalar<f32>>} : () -> !rise.scalar<f32>
          %19 = "rise.reduceSeq"() {n = #rise.nat<4>, s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<4, scalar<f32>> -> scalar<f32>>>>
          %20 = "rise.apply"(%19, %17, %18, %16) : (!rise.fun<fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<4, scalar<f32>> -> scalar<f32>>>>, !rise.fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>>, !rise.scalar<f32>, !rise.array<4, scalar<f32>>) -> !rise.scalar<f32>
          "rise.return"(%20) : (!rise.scalar<f32>) -> ()
        }) : () -> !rise.fun<array<4, scalar<f32>> -> scalar<f32>>
        %10 = "rise.mapSeq"() {n = #rise.nat<4>, s = #rise.array<4, scalar<f32>>, t = #rise.scalar<f32>} : () -> !rise.fun<fun<array<4, scalar<f32>> -> scalar<f32>> -> fun<array<4, array<4, scalar<f32>>> -> array<4, scalar<f32>>>>
        %11 = "rise.apply"(%10, %9, %1) : (!rise.fun<fun<array<4, scalar<f32>> -> scalar<f32>> -> fun<array<4, array<4, scalar<f32>>> -> array<4, scalar<f32>>>>, !rise.fun<array<4, scalar<f32>> -> scalar<f32>>, !rise.array<4, array<4, scalar<f32>>>) -> !rise.array<4, scalar<f32>>
        "rise.return"(%11) : (!rise.array<4, scalar<f32>>) -> ()
      }) : () -> !rise.fun<array<4, scalar<f32>> -> array<4, scalar<f32>>>
      %3 = "rise.mapSeq"() {n = #rise.nat<4>, s = #rise.array<4, scalar<f32>>, t = #rise.array<4, scalar<f32>>} : () -> !rise.fun<fun<array<4, scalar<f32>> -> array<4, scalar<f32>>> -> fun<array<4, array<4, scalar<f32>>> -> array<4, array<4, scalar<f32>>>>>
      %4 = "rise.apply"(%3, %2, %0) : (!rise.fun<fun<array<4, scalar<f32>> -> array<4, scalar<f32>>> -> fun<array<4, array<4, scalar<f32>>> -> array<4, array<4, scalar<f32>>>>>, !rise.fun<array<4, scalar<f32>> -> array<4, scalar<f32>>>, !rise.array<4, array<4, scalar<f32>>>) -> !rise.array<4, array<4, scalar<f32>>>
      %5 = "rise.transpose"() {m = #rise.nat<4>, n = #rise.nat<4>, t = #rise.scalar<f32>} : () -> !rise.fun<array<4, array<4, scalar<f32>>> -> array<4, array<4, scalar<f32>>>>
      %6 = "rise.apply"(%5, %4) : (!rise.fun<array<4, array<4, scalar<f32>>> -> array<4, array<4, scalar<f32>>>>, !rise.array<4, array<4, scalar<f32>>>) -> !rise.array<4, array<4, scalar<f32>>>
      %7 = "rise.transpose"() {m = #rise.nat<4>, n = #rise.nat<4>, t = #rise.scalar<f32>} : () -> !rise.fun<array<4, array<4, scalar<f32>>> -> array<4, array<4, scalar<f32>>>>
      %8 = "rise.apply"(%7, %6) : (!rise.fun<array<4, array<4, scalar<f32>>> -> array<4, array<4, scalar<f32>>>>, !rise.array<4, array<4, scalar<f32>>>) -> !rise.array<4, array<4, scalar<f32>>>
      "rise.out"(%arg0, %8) : (memref<4x4xf32>, !rise.array<4, array<4, scalar<f32>>>) -> ()
      "rise.return"() : () -> ()
    }) : () -> ()
    return
  }
  func @mm() {
    %0 = alloc() : memref<4x4xf32>
    %1 = alloc() : memref<4x4xf32>
    %cst = constant 5.000000e+00 : f32
    linalg.fill(%1, %cst) : memref<4x4xf32>, f32
    %2 = alloc() : memref<4x4xf32>
    %cst_0 = constant 5.000000e+00 : f32
    linalg.fill(%2, %cst_0) : memref<4x4xf32>, f32
    call @rise_fun(%0, %1, %2) : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
    %3 = memref_cast %0 : memref<4x4xf32> to memref<*xf32>
    call @print_memref_f32(%3) : (memref<*xf32>) -> ()
    return
  }
}
// MM: Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 4] strides = [4, 1] data =
// MM: {{[[100,   100,   100,   100],}}
// MM: [100,   100,   100,   100],
// MM: [100,   100,   100,   100],
// MM: [100,   100,   100,   100]]