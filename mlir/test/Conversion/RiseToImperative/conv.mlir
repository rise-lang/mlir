// mlir-opt %s
// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e conv2D_test -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext,/opt/intel/lib/intel64_lin/libmkl_intel_ilp64.so,/home/martin/development/phd/projects/MLIR/performance_measuring/dylib/measure_lib.so | FileCheck %s --check-prefix=MM_irreg


func @print_memref_f32(memref<*xf32>)
func @rtclock() -> (f64)
func @print_time(f64,f64)

func @conv2D(%arg0: memref<7x7xf32>, %arg1: memref<3x3xf32>, %arg2: memref<5x5xf32>) {
  "rise.lowering_unit"() ( {
    %0 = "rise.in"(%arg0) : (memref<7x7xf32>) -> !rise.array<7, array<7, scalar<f32>>>
    %1 = "rise.in"(%arg1) : (memref<3x3xf32>) -> !rise.array<3, array<3, scalar<f32>>>
    %2 = "rise.lambda"() ( {
    ^bb0(%arg3: !rise.array<7, scalar<f32>>):  // no predecessors
      %13 = "rise.slide"() {n = #rise.nat<5>, sp = #rise.nat<1>, sz = #rise.nat<3>, t = #rise.scalar<f32>} : () -> !rise.fun<array<7, scalar<f32>> -> array<5, array<3, scalar<f32>>>>
      %14 = "rise.apply"(%13, %arg3) : (!rise.fun<array<7, scalar<f32>> -> array<5, array<3, scalar<f32>>>>, !rise.array<7, scalar<f32>>) -> !rise.array<5, array<3, scalar<f32>>>
      "rise.return"(%14) : (!rise.array<5, array<3, scalar<f32>>>) -> ()
    }) : () -> !rise.fun<array<7, scalar<f32>> -> array<5, array<3, scalar<f32>>>>
    %3 = "rise.map"() {n = #rise.nat<7>, s = #rise.array<7, scalar<f32>>, t = #rise.array<5, array<3, scalar<f32>>>} : () -> !rise.fun<fun<array<7, scalar<f32>> -> array<5, array<3, scalar<f32>>>> -> fun<array<7, array<7, scalar<f32>>> -> array<7, array<5, array<3, scalar<f32>>>>>>
    %4 = "rise.apply"(%3, %2, %0) : (!rise.fun<fun<array<7, scalar<f32>> -> array<5, array<3, scalar<f32>>>> -> fun<array<7, array<7, scalar<f32>>> -> array<7, array<5, array<3, scalar<f32>>>>>>, !rise.fun<array<7, scalar<f32>> -> array<5, array<3, scalar<f32>>>>, !rise.array<7, array<7, scalar<f32>>>) -> !rise.array<7, array<5, array<3, scalar<f32>>>>
    %5 = "rise.slide"() {n = #rise.nat<5>, sp = #rise.nat<1>, sz = #rise.nat<3>, t = #rise.array<5, array<3, scalar<f32>>>} : () -> !rise.fun<array<7, array<5, array<3, scalar<f32>>>> -> array<5, array<3, array<5, array<3, scalar<f32>>>>>>
    %6 = "rise.apply"(%5, %4) : (!rise.fun<array<7, array<5, array<3, scalar<f32>>>> -> array<5, array<3, array<5, array<3, scalar<f32>>>>>>, !rise.array<7, array<5, array<3, scalar<f32>>>>) -> !rise.array<5, array<3, array<5, array<3, scalar<f32>>>>>
    %7 = "rise.lambda"() ( {
    ^bb0(%arg3: !rise.array<3, array<5, array<3, scalar<f32>>>>):  // no predecessors
      %13 = "rise.transpose"() {m = #rise.nat<5>, n = #rise.nat<3>, t = #rise.array<3, scalar<f32>>} : () -> !rise.fun<array<3, array<5, array<3, scalar<f32>>>> -> array<5, array<3, array<3, scalar<f32>>>>>
      %14 = "rise.apply"(%13, %arg3) : (!rise.fun<array<3, array<5, array<3, scalar<f32>>>> -> array<5, array<3, array<3, scalar<f32>>>>>, !rise.array<3, array<5, array<3, scalar<f32>>>>) -> !rise.array<5, array<3, array<3, scalar<f32>>>>
      "rise.return"(%14) : (!rise.array<5, array<3, array<3, scalar<f32>>>>) -> ()
    }) : () -> !rise.fun<array<3, array<5, array<3, scalar<f32>>>> -> array<5, array<3, array<3, scalar<f32>>>>>
    %8 = "rise.map"() {n = #rise.nat<5>, s = #rise.array<3, array<5, array<3, scalar<f32>>>>, t = #rise.array<5, array<3, array<3, scalar<f32>>>>} : () -> !rise.fun<fun<array<3, array<5, array<3, scalar<f32>>>> -> array<5, array<3, array<3, scalar<f32>>>>> -> fun<array<5, array<3, array<5, array<3, scalar<f32>>>>> -> array<5, array<5, array<3, array<3, scalar<f32>>>>>>>
    %9 = "rise.apply"(%8, %7, %6) : (!rise.fun<fun<array<3, array<5, array<3, scalar<f32>>>> -> array<5, array<3, array<3, scalar<f32>>>>> -> fun<array<5, array<3, array<5, array<3, scalar<f32>>>>> -> array<5, array<5, array<3, array<3, scalar<f32>>>>>>>, !rise.fun<array<3, array<5, array<3, scalar<f32>>>> -> array<5, array<3, array<3, scalar<f32>>>>>, !rise.array<5, array<3, array<5, array<3, scalar<f32>>>>>) -> !rise.array<5, array<5, array<3, array<3, scalar<f32>>>>>
    %10 = "rise.lambda"() ( {
    ^bb0(%arg3: !rise.array<5, array<3, array<3, scalar<f32>>>>):  // no predecessors
      %13 = "rise.lambda"() ( {
      ^bb0(%arg4: !rise.array<3, array<3, scalar<f32>>>):  // no predecessors
        %16 = "rise.zip"() {n = #rise.nat<3>, s = #rise.array<3, scalar<f32>>, t = #rise.array<3, scalar<f32>>} : () -> !rise.fun<array<3, array<3, scalar<f32>>> -> fun<array<3, array<3, scalar<f32>>> -> array<3, tuple<array<3, scalar<f32>>, array<3, scalar<f32>>>>>>
        %17 = "rise.apply"(%16, %arg4, %1) : (!rise.fun<array<3, array<3, scalar<f32>>> -> fun<array<3, array<3, scalar<f32>>> -> array<3, tuple<array<3, scalar<f32>>, array<3, scalar<f32>>>>>>, !rise.array<3, array<3, scalar<f32>>>, !rise.array<3, array<3, scalar<f32>>>) -> !rise.array<3, tuple<array<3, scalar<f32>>, array<3, scalar<f32>>>>
        %18 = "rise.lambda"() ( {
        ^bb0(%arg5: !rise.tuple<array<3, scalar<f32>>, array<3, scalar<f32>>>):  // no predecessors
          %27 = "rise.snd"() {s = #rise.array<3, scalar<f32>>, t = #rise.array<3, scalar<f32>>} : () -> !rise.fun<tuple<array<3, scalar<f32>>, array<3, scalar<f32>>> -> array<3, scalar<f32>>>
          %28 = "rise.apply"(%27, %arg5) : (!rise.fun<tuple<array<3, scalar<f32>>, array<3, scalar<f32>>> -> array<3, scalar<f32>>>, !rise.tuple<array<3, scalar<f32>>, array<3, scalar<f32>>>) -> !rise.array<3, scalar<f32>>
          %29 = "rise.fst"() {s = #rise.array<3, scalar<f32>>, t = #rise.array<3, scalar<f32>>} : () -> !rise.fun<tuple<array<3, scalar<f32>>, array<3, scalar<f32>>> -> array<3, scalar<f32>>>
          %30 = "rise.apply"(%29, %arg5) : (!rise.fun<tuple<array<3, scalar<f32>>, array<3, scalar<f32>>> -> array<3, scalar<f32>>>, !rise.tuple<array<3, scalar<f32>>, array<3, scalar<f32>>>) -> !rise.array<3, scalar<f32>>
          %31 = "rise.zip"() {n = #rise.nat<3>, s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<array<3, scalar<f32>> -> fun<array<3, scalar<f32>> -> array<3, tuple<scalar<f32>, scalar<f32>>>>>
          %32 = "rise.apply"(%31, %30, %28) : (!rise.fun<array<3, scalar<f32>> -> fun<array<3, scalar<f32>> -> array<3, tuple<scalar<f32>, scalar<f32>>>>>, !rise.array<3, scalar<f32>>, !rise.array<3, scalar<f32>>) -> !rise.array<3, tuple<scalar<f32>, scalar<f32>>>
          "rise.return"(%32) : (!rise.array<3, tuple<scalar<f32>, scalar<f32>>>) -> ()
        }) : () -> !rise.fun<tuple<array<3, scalar<f32>>, array<3, scalar<f32>>> -> array<3, tuple<scalar<f32>, scalar<f32>>>>
        %19 = "rise.map"() {n = #rise.nat<3>, s = #rise.tuple<array<3, scalar<f32>>, array<3, scalar<f32>>>, t = #rise.array<3, tuple<scalar<f32>, scalar<f32>>>} : () -> !rise.fun<fun<tuple<array<3, scalar<f32>>, array<3, scalar<f32>>> -> array<3, tuple<scalar<f32>, scalar<f32>>>> -> fun<array<3, tuple<array<3, scalar<f32>>, array<3, scalar<f32>>>> -> array<3, array<3, tuple<scalar<f32>, scalar<f32>>>>>>
        %20 = "rise.apply"(%19, %18, %17) : (!rise.fun<fun<tuple<array<3, scalar<f32>>, array<3, scalar<f32>>> -> array<3, tuple<scalar<f32>, scalar<f32>>>> -> fun<array<3, tuple<array<3, scalar<f32>>, array<3, scalar<f32>>>> -> array<3, array<3, tuple<scalar<f32>, scalar<f32>>>>>>, !rise.fun<tuple<array<3, scalar<f32>>, array<3, scalar<f32>>> -> array<3, tuple<scalar<f32>, scalar<f32>>>>, !rise.array<3, tuple<array<3, scalar<f32>>, array<3, scalar<f32>>>>) -> !rise.array<3, array<3, tuple<scalar<f32>, scalar<f32>>>>
        %21 = "rise.join"() {m = #rise.nat<3>, n = #rise.nat<3>, t = #rise.tuple<scalar<f32>, scalar<f32>>} : () -> !rise.fun<array<3, array<3, tuple<scalar<f32>, scalar<f32>>>> -> array<9, tuple<scalar<f32>, scalar<f32>>>>
        %22 = "rise.apply"(%21, %20) : (!rise.fun<array<3, array<3, tuple<scalar<f32>, scalar<f32>>>> -> array<9, tuple<scalar<f32>, scalar<f32>>>>, !rise.array<3, array<3, tuple<scalar<f32>, scalar<f32>>>>) -> !rise.array<9, tuple<scalar<f32>, scalar<f32>>>
        %23 = "rise.literal"() {literal = #rise.lit<0.000000, scalar<f32>>} : () -> !rise.scalar<f32>
        %24 = "rise.lambda"() ( {
        ^bb0(%arg5: !rise.tuple<scalar<f32>, scalar<f32>>, %arg6: !rise.scalar<f32>):  // no predecessors
          %27 = "rise.fst"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
          %28 = "rise.apply"(%27, %arg5) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
          %29 = "rise.snd"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
          %30 = "rise.apply"(%29, %arg5) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
          %31 = "rise.embed"(%28, %30, %arg6) ( {
          ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):  // no predecessors
            %32 = mulf %arg7, %arg8 : f32
            %33 = addf %arg9, %32 : f32
            "rise.return"(%33) : (f32) -> ()
          }) : (!rise.scalar<f32>, !rise.scalar<f32>, !rise.scalar<f32>) -> !rise.scalar<f32>
          "rise.return"(%31) : (!rise.scalar<f32>) -> ()
        }) : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>
        %25 = "rise.reduceSeq"() {n = #rise.nat<9>, s = #rise.tuple<scalar<f32>, scalar<f32>>, t = #rise.scalar<f32>, to = "scf"} : () -> !rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<9, tuple<scalar<f32>, scalar<f32>>> -> scalar<f32>>>>
        %26 = "rise.apply"(%25, %24, %23, %22) : (!rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<9, tuple<scalar<f32>, scalar<f32>>> -> scalar<f32>>>>, !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>, !rise.scalar<f32>, !rise.array<9, tuple<scalar<f32>, scalar<f32>>>) -> !rise.scalar<f32>
        "rise.return"(%26) : (!rise.scalar<f32>) -> ()
      }) : () -> !rise.fun<array<3, array<3, scalar<f32>>> -> scalar<f32>>
      %14 = "rise.mapSeq"() {n = #rise.nat<5>, s = #rise.array<3, array<3, scalar<f32>>>, t = #rise.scalar<f32>, to = "scf"} : () -> !rise.fun<fun<array<3, array<3, scalar<f32>>> -> scalar<f32>> -> fun<array<5, array<3, array<3, scalar<f32>>>> -> array<5, scalar<f32>>>>
      %15 = "rise.apply"(%14, %13, %arg3) : (!rise.fun<fun<array<3, array<3, scalar<f32>>> -> scalar<f32>> -> fun<array<5, array<3, array<3, scalar<f32>>>> -> array<5, scalar<f32>>>>, !rise.fun<array<3, array<3, scalar<f32>>> -> scalar<f32>>, !rise.array<5, array<3, array<3, scalar<f32>>>>) -> !rise.array<5, scalar<f32>>
      "rise.return"(%15) : (!rise.array<5, scalar<f32>>) -> ()
    }) : () -> !rise.fun<array<5, array<3, array<3, scalar<f32>>>> -> array<5, scalar<f32>>>
    %11 = "rise.mapSeq"() {n = #rise.nat<5>, s = #rise.array<5, array<3, array<3, scalar<f32>>>>, t = #rise.array<5, scalar<f32>>, to = "scf"} : () -> !rise.fun<fun<array<5, array<3, array<3, scalar<f32>>>> -> array<5, scalar<f32>>> -> fun<array<5, array<5, array<3, array<3, scalar<f32>>>>> -> array<5, array<5, scalar<f32>>>>>
    %12 = "rise.apply"(%11, %10, %9) : (!rise.fun<fun<array<5, array<3, array<3, scalar<f32>>>> -> array<5, scalar<f32>>> -> fun<array<5, array<5, array<3, array<3, scalar<f32>>>>> -> array<5, array<5, scalar<f32>>>>>, !rise.fun<array<5, array<3, array<3, scalar<f32>>>> -> array<5, scalar<f32>>>, !rise.array<5, array<5, array<3, array<3, scalar<f32>>>>>) -> !rise.array<5, array<5, scalar<f32>>>
    "rise.out"(%arg2, %12) : (memref<5x5xf32>, !rise.array<5, array<5, scalar<f32>>>) -> ()
    "rise.return"() : () -> ()
  }) : () -> ()
  return
}
func @conv2D_test() {
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c1 = constant 1 : index
  %c0_0 = constant 0 : index
  %c3_1 = constant 3 : index
  %c1_2 = constant 1 : index
  %cst = constant 1.000000e+00 : f32
  %0 = alloc() : memref<3x3xf32>
  scf.for %arg0 = %c0 to %c3 step %c1 {
    scf.for %arg1 = %c0_0 to %c3_1 step %c1_2 {
      store %cst, %0[%arg0, %arg1] : memref<3x3xf32>
    }
  }
  %c0_3 = constant 0 : index
  %c7 = constant 7 : index
  %c1_4 = constant 1 : index
  %c0_5 = constant 0 : index
  %c7_6 = constant 7 : index
  %c1_7 = constant 1 : index
  %cst_8 = constant 0.000000e+00 : f32
  %cst_9 = constant 1.000000e+00 : f32
  %1 = alloc() : memref<f32>
  store %cst_8, %1[] : memref<f32>
  %2 = alloc() : memref<7x7xf32>
  scf.for %arg0 = %c0_3 to %c7 step %c1_4 {
    scf.for %arg1 = %c0_5 to %c7_6 step %c1_7 {
      %7 = load %1[] : memref<f32>
      store %7, %2[%arg0, %arg1] : memref<7x7xf32>
      %8 = load %1[] : memref<f32>
      %9 = addf %8, %cst_9 : f32
      store %9, %1[] : memref<f32>
    }
  }
  %3 = alloc() : memref<5x5xf32>

      %t0 = call @rtclock() : () -> (f64)
  call @conv2D(%2, %0, %3) : (memref<7x7xf32>, memref<3x3xf32>, memref<5x5xf32>) -> ()
        %t1 = call @rtclock() : () -> (f64)
  %4 = memref_cast %2 : memref<7x7xf32> to memref<*xf32>
  call @print_memref_f32(%4) : (memref<*xf32>) -> ()
  %5 = memref_cast %0 : memref<3x3xf32> to memref<*xf32>
  call @print_memref_f32(%5) : (memref<*xf32>) -> ()
  %6 = memref_cast %3 : memref<5x5xf32> to memref<*xf32>
  call @print_memref_f32(%6) : (memref<*xf32>) -> ()

     call @print_time(%t0, %t1): (f64,f64) -> ()

  return
}

