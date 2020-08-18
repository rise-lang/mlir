// mlir-opt %s
// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e conv2DSeparable_test -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext,/opt/intel/lib/intel64_lin/libmkl_intel_ilp64.so,/home/martin/development/phd/projects/MLIR/performance_measuring/dylib/measure_lib.so | FileCheck %s --check-prefix=MM_irreg


func @print_memref_f32(memref<*xf32>)
func @rtclock() -> (f64)
func @print_time(f64,f64)

func @conv2DSeparable(%arg0: memref<9x9xf32>, %arg1: memref<3xf32>, %arg2: memref<3xf32>, %arg3: memref<9x9xf32>) {
  "rise.lowering_unit"() ( {
    %0 = "rise.in"(%arg0) : (memref<9x9xf32>) -> !rise.array<9, array<9, scalar<f32>>>
    %1 = "rise.in"(%arg1) : (memref<3xf32>) -> !rise.array<3, scalar<f32>>
    %2 = "rise.in"(%arg2) : (memref<3xf32>) -> !rise.array<3, scalar<f32>>
    %3 = "rise.lambda"() ( {
    ^bb0(%arg4: !rise.array<9, scalar<f32>>):  // no predecessors
      %16 = "rise.pad"() {l = #rise.nat<1>, n = #rise.nat<9>, r = #rise.nat<1>, t = #rise.scalar<f32>} : () -> !rise.fun<array<9, scalar<f32>> -> array<11, scalar<f32>>>
      %17 = "rise.apply"(%16, %arg4) : (!rise.fun<array<9, scalar<f32>> -> array<11, scalar<f32>>>, !rise.array<9, scalar<f32>>) -> !rise.array<11, scalar<f32>>
      "rise.return"(%17) : (!rise.array<11, scalar<f32>>) -> ()
    }) : () -> !rise.fun<array<9, scalar<f32>> -> array<11, scalar<f32>>>
    %4 = "rise.map"() {n = #rise.nat<9>, s = #rise.array<9, scalar<f32>>, t = #rise.array<11, scalar<f32>>} : () -> !rise.fun<fun<array<9, scalar<f32>> -> array<11, scalar<f32>>> -> fun<array<9, array<9, scalar<f32>>> -> array<9, array<11, scalar<f32>>>>>
    %5 = "rise.apply"(%4, %3, %0) : (!rise.fun<fun<array<9, scalar<f32>> -> array<11, scalar<f32>>> -> fun<array<9, array<9, scalar<f32>>> -> array<9, array<11, scalar<f32>>>>>, !rise.fun<array<9, scalar<f32>> -> array<11, scalar<f32>>>, !rise.array<9, array<9, scalar<f32>>>) -> !rise.array<9, array<11, scalar<f32>>>
    %6 = "rise.pad"() {l = #rise.nat<1>, n = #rise.nat<9>, r = #rise.nat<1>, t = #rise.array<11, scalar<f32>>} : () -> !rise.fun<array<9, array<11, scalar<f32>>> -> array<11, array<11, scalar<f32>>>>
    %7 = "rise.apply"(%6, %5) : (!rise.fun<array<9, array<11, scalar<f32>>> -> array<11, array<11, scalar<f32>>>>, !rise.array<9, array<11, scalar<f32>>>) -> !rise.array<11, array<11, scalar<f32>>>
    %8 = "rise.slide"() {n = #rise.nat<9>, sp = #rise.nat<1>, sz = #rise.nat<3>, t = #rise.array<11, scalar<f32>>} : () -> !rise.fun<array<11, array<11, scalar<f32>>> -> array<9, array<3, array<11, scalar<f32>>>>>
    %9 = "rise.apply"(%8, %7) : (!rise.fun<array<11, array<11, scalar<f32>>> -> array<9, array<3, array<11, scalar<f32>>>>>, !rise.array<11, array<11, scalar<f32>>>) -> !rise.array<9, array<3, array<11, scalar<f32>>>>
    %10 = "rise.lambda"() ( {
    ^bb0(%arg4: !rise.array<3, array<11, scalar<f32>>>):  // no predecessors
      %16 = "rise.transpose"() {m = #rise.nat<11>, n = #rise.nat<3>, t = #rise.scalar<f32>} : () -> !rise.fun<array<3, array<11, scalar<f32>>> -> array<11, array<3, scalar<f32>>>>
      %17 = "rise.apply"(%16, %arg4) : (!rise.fun<array<3, array<11, scalar<f32>>> -> array<11, array<3, scalar<f32>>>>, !rise.array<3, array<11, scalar<f32>>>) -> !rise.array<11, array<3, scalar<f32>>>
      %18 = "rise.lambda"() ( {
      ^bb0(%arg5: !rise.array<3, scalar<f32>>):  // no predecessors
        %21 = "rise.zip"() {n = #rise.nat<3>, s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<array<3, scalar<f32>> -> fun<array<3, scalar<f32>> -> array<3, tuple<scalar<f32>, scalar<f32>>>>>
        %22 = "rise.apply"(%21, %arg5, %2) : (!rise.fun<array<3, scalar<f32>> -> fun<array<3, scalar<f32>> -> array<3, tuple<scalar<f32>, scalar<f32>>>>>, !rise.array<3, scalar<f32>>, !rise.array<3, scalar<f32>>) -> !rise.array<3, tuple<scalar<f32>, scalar<f32>>>
        %23 = "rise.literal"() {literal = #rise.lit<0.000000, scalar<f32>>} : () -> !rise.scalar<f32>
        %24 = "rise.lambda"() ( {
        ^bb0(%arg6: !rise.tuple<scalar<f32>, scalar<f32>>, %arg7: !rise.scalar<f32>):  // no predecessors
          %27 = "rise.fst"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
          %28 = "rise.apply"(%27, %arg6) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
          %29 = "rise.snd"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
          %30 = "rise.apply"(%29, %arg6) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
          %31 = "rise.embed"(%28, %30, %arg7) ( {
          ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
            %32 = mulf %arg8, %arg9 : f32
            %33 = addf %arg10, %32 : f32
            "rise.return"(%33) : (f32) -> ()
          }) : (!rise.scalar<f32>, !rise.scalar<f32>, !rise.scalar<f32>) -> !rise.scalar<f32>
          "rise.return"(%31) : (!rise.scalar<f32>) -> ()
        }) : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>
        %25 = "rise.reduceSeq"() {n = #rise.nat<3>, s = #rise.tuple<scalar<f32>, scalar<f32>>, t = #rise.scalar<f32>, to = "scf"} : () -> !rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<3, tuple<scalar<f32>, scalar<f32>>> -> scalar<f32>>>>
        %26 = "rise.apply"(%25, %24, %23, %22) : (!rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<3, tuple<scalar<f32>, scalar<f32>>> -> scalar<f32>>>>, !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>, !rise.scalar<f32>, !rise.array<3, tuple<scalar<f32>, scalar<f32>>>) -> !rise.scalar<f32>
        "rise.return"(%26) : (!rise.scalar<f32>) -> ()
      }) : () -> !rise.fun<array<3, scalar<f32>> -> scalar<f32>>
      %19 = "rise.mapSeq"() {n = #rise.nat<11>, s = #rise.array<3, scalar<f32>>, t = #rise.scalar<f32>, to = "scf"} : () -> !rise.fun<fun<array<3, scalar<f32>> -> scalar<f32>> -> fun<array<11, array<3, scalar<f32>>> -> array<11, scalar<f32>>>>
      %20 = "rise.apply"(%19, %18, %17) : (!rise.fun<fun<array<3, scalar<f32>> -> scalar<f32>> -> fun<array<11, array<3, scalar<f32>>> -> array<11, scalar<f32>>>>, !rise.fun<array<3, scalar<f32>> -> scalar<f32>>, !rise.array<11, array<3, scalar<f32>>>) -> !rise.array<11, scalar<f32>>
      "rise.return"(%20) : (!rise.array<11, scalar<f32>>) -> ()
    }) : () -> !rise.fun<array<3, array<11, scalar<f32>>> -> array<11, scalar<f32>>>
    %11 = "rise.mapSeq"() {n = #rise.nat<9>, s = #rise.array<3, array<11, scalar<f32>>>, t = #rise.array<11, scalar<f32>>, to = "scf"} : () -> !rise.fun<fun<array<3, array<11, scalar<f32>>> -> array<11, scalar<f32>>> -> fun<array<9, array<3, array<11, scalar<f32>>>> -> array<9, array<11, scalar<f32>>>>>
    %12 = "rise.apply"(%11, %10, %9) : (!rise.fun<fun<array<3, array<11, scalar<f32>>> -> array<11, scalar<f32>>> -> fun<array<9, array<3, array<11, scalar<f32>>>> -> array<9, array<11, scalar<f32>>>>>, !rise.fun<array<3, array<11, scalar<f32>>> -> array<11, scalar<f32>>>, !rise.array<9, array<3, array<11, scalar<f32>>>>) -> !rise.array<9, array<11, scalar<f32>>>
    %13 = "rise.lambda"() ( {
    ^bb0(%arg4: !rise.array<11, scalar<f32>>):  // no predecessors
      %16 = "rise.slide"() {n = #rise.nat<9>, sp = #rise.nat<1>, sz = #rise.nat<3>, t = #rise.scalar<f32>} : () -> !rise.fun<array<11, scalar<f32>> -> array<9, array<3, scalar<f32>>>>
      %17 = "rise.apply"(%16, %arg4) : (!rise.fun<array<11, scalar<f32>> -> array<9, array<3, scalar<f32>>>>, !rise.array<11, scalar<f32>>) -> !rise.array<9, array<3, scalar<f32>>>
      %18 = "rise.lambda"() ( {
      ^bb0(%arg5: !rise.array<3, scalar<f32>>):  // no predecessors
        %21 = "rise.zip"() {n = #rise.nat<3>, s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<array<3, scalar<f32>> -> fun<array<3, scalar<f32>> -> array<3, tuple<scalar<f32>, scalar<f32>>>>>
        %22 = "rise.apply"(%21, %arg5, %1) : (!rise.fun<array<3, scalar<f32>> -> fun<array<3, scalar<f32>> -> array<3, tuple<scalar<f32>, scalar<f32>>>>>, !rise.array<3, scalar<f32>>, !rise.array<3, scalar<f32>>) -> !rise.array<3, tuple<scalar<f32>, scalar<f32>>>
        %23 = "rise.literal"() {literal = #rise.lit<0.000000, scalar<f32>>} : () -> !rise.scalar<f32>
        %24 = "rise.lambda"() ( {
        ^bb0(%arg6: !rise.tuple<scalar<f32>, scalar<f32>>, %arg7: !rise.scalar<f32>):  // no predecessors
          %27 = "rise.fst"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
          %28 = "rise.apply"(%27, %arg6) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
          %29 = "rise.snd"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
          %30 = "rise.apply"(%29, %arg6) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
          %31 = "rise.embed"(%28, %30, %arg7) ( {
          ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
            %32 = mulf %arg8, %arg9 : f32
            %33 = addf %arg10, %32 : f32
            "rise.return"(%33) : (f32) -> ()
          }) : (!rise.scalar<f32>, !rise.scalar<f32>, !rise.scalar<f32>) -> !rise.scalar<f32>
          "rise.return"(%31) : (!rise.scalar<f32>) -> ()
        }) : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>
        %25 = "rise.reduceSeq"() {n = #rise.nat<3>, s = #rise.tuple<scalar<f32>, scalar<f32>>, t = #rise.scalar<f32>, to = "scf"} : () -> !rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<3, tuple<scalar<f32>, scalar<f32>>> -> scalar<f32>>>>
        %26 = "rise.apply"(%25, %24, %23, %22) : (!rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<3, tuple<scalar<f32>, scalar<f32>>> -> scalar<f32>>>>, !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>, !rise.scalar<f32>, !rise.array<3, tuple<scalar<f32>, scalar<f32>>>) -> !rise.scalar<f32>
        "rise.return"(%26) : (!rise.scalar<f32>) -> ()
      }) : () -> !rise.fun<array<3, scalar<f32>> -> scalar<f32>>
      %19 = "rise.mapSeq"() {n = #rise.nat<9>, s = #rise.array<3, scalar<f32>>, t = #rise.scalar<f32>, to = "scf"} : () -> !rise.fun<fun<array<3, scalar<f32>> -> scalar<f32>> -> fun<array<9, array<3, scalar<f32>>> -> array<9, scalar<f32>>>>
      %20 = "rise.apply"(%19, %18, %17) : (!rise.fun<fun<array<3, scalar<f32>> -> scalar<f32>> -> fun<array<9, array<3, scalar<f32>>> -> array<9, scalar<f32>>>>, !rise.fun<array<3, scalar<f32>> -> scalar<f32>>, !rise.array<9, array<3, scalar<f32>>>) -> !rise.array<9, scalar<f32>>
      "rise.return"(%20) : (!rise.array<9, scalar<f32>>) -> ()
    }) : () -> !rise.fun<array<11, scalar<f32>> -> array<9, scalar<f32>>>
    %14 = "rise.mapSeq"() {n = #rise.nat<9>, s = #rise.array<11, scalar<f32>>, t = #rise.array<9, scalar<f32>>, to = "scf"} : () -> !rise.fun<fun<array<11, scalar<f32>> -> array<9, scalar<f32>>> -> fun<array<9, array<11, scalar<f32>>> -> array<9, array<9, scalar<f32>>>>>
    %15 = "rise.apply"(%14, %13, %12) : (!rise.fun<fun<array<11, scalar<f32>> -> array<9, scalar<f32>>> -> fun<array<9, array<11, scalar<f32>>> -> array<9, array<9, scalar<f32>>>>>, !rise.fun<array<11, scalar<f32>> -> array<9, scalar<f32>>>, !rise.array<9, array<11, scalar<f32>>>) -> !rise.array<9, array<9, scalar<f32>>>
    "rise.out"(%arg3, %15) : (memref<9x9xf32>, !rise.array<9, array<9, scalar<f32>>>) -> ()
    "rise.return"() : () -> ()
  }) : () -> ()
  return
}func @conv2DSeparable_test() {
  %c0 = constant 0 : index
  %c3 = constant 3 : index
  %c1 = constant 1 : index
  %cst = constant 1.000000e+00 : f32
  %0 = alloc() : memref<3xf32>
  scf.for %arg0 = %c0 to %c3 step %c1 {
    store %cst, %0[%arg0] : memref<3xf32>
  }
  %c0_0 = constant 0 : index
  %c3_1 = constant 3 : index
  %c1_2 = constant 1 : index
  %cst_3 = constant 1.000000e+00 : f32
  %1 = alloc() : memref<3xf32>
  scf.for %arg0 = %c0_0 to %c3_1 step %c1_2 {
    store %cst_3, %1[%arg0] : memref<3xf32>
  }
  %c0_4 = constant 0 : index
  %c9 = constant 9 : index
  %c1_5 = constant 1 : index
  %c0_6 = constant 0 : index
  %c9_7 = constant 9 : index
  %c1_8 = constant 1 : index
  %cst_9 = constant 0.000000e+00 : f32
  %cst_10 = constant 1.000000e+00 : f32
  %2 = alloc() : memref<f32>
  store %cst_9, %2[] : memref<f32>
  %3 = alloc() : memref<9x9xf32>
  scf.for %arg0 = %c0_4 to %c9 step %c1_5 {
    scf.for %arg1 = %c0_6 to %c9_7 step %c1_8 {
      %11 = load %2[] : memref<f32>
      store %11, %3[%arg0, %arg1] : memref<9x9xf32>
      %12 = load %2[] : memref<f32>
      %13 = addf %12, %cst_10 : f32
      store %13, %2[] : memref<f32>
    }
  }
  %4 = alloc() : memref<9x9xf32>
  %5 = call @rtclock() : () -> f64
  call @conv2DSeparable(%3, %1, %0, %4) : (memref<9x9xf32>, memref<3xf32>, memref<3xf32>, memref<9x9xf32>) -> ()
  %6 = call @rtclock() : () -> f64
  call @print_time(%5, %6) : (f64, f64) -> ()
  %7 = memref_cast %3 : memref<9x9xf32> to memref<*xf32>
  call @print_memref_f32(%7) : (memref<*xf32>) -> ()
  %8 = memref_cast %1 : memref<3xf32> to memref<*xf32>
  call @print_memref_f32(%8) : (memref<*xf32>) -> ()
  %9 = memref_cast %0 : memref<3xf32> to memref<*xf32>
  call @print_memref_f32(%9) : (memref<*xf32>) -> ()
  %10 = memref_cast %4 : memref<9x9xf32> to memref<*xf32>
  call @print_memref_f32(%10) : (memref<*xf32>) -> ()
  return
}

