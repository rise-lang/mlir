// RUN: mlir-opt %s
// RU: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e test -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=TEST

//define n = 2
//  %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<1x5x5x1xf32>
//  %c0 = constant 0 : index
//  %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<1x7x7x1xf32>
//  %2 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1} : memref<3x3x1x1xf32>
func @print_memref_f32(memref<*xf32>)
func @print_f32(f32)
func @print_bin_op(f32, f32, f32)
func @conv2DSeparable(%arg0: memref<9x9xf32>, %arg1: memref<3xf32>, %arg2: memref<3xf32>, %arg3: memref<9x9xf32>) {
  "rise.lowering_unit"() ( {
           %0 = "rise.in"(%arg0) : (memref<9x9xf32>) -> !rise.array<9, array<9, scalar<f32>>>
            %1 = "rise.in"(%arg1) : (memref<3xf32>) -> !rise.array<3, scalar<f32>>
            %2 = "rise.in"(%arg2) : (memref<3xf32>) -> !rise.array<3, scalar<f32>>
            %3 = "rise.lambda"() ( {
            ^bb0(%arg4: !rise.array<9, scalar<f32>>):  // no predecessors
              %11 = "rise.pad"() {l = #rise.nat<1>, n = #rise.nat<9>, r = #rise.nat<1>, t = #rise.scalar<f32>} : () -> !rise.fun<array<9, scalar<f32>> -> array<11, scalar<f32>>>
              %12 = "rise.apply"(%11, %arg4) : (!rise.fun<array<9, scalar<f32>> -> array<11, scalar<f32>>>, !rise.array<9, scalar<f32>>) -> !rise.array<11, scalar<f32>>
              "rise.return"(%12) : (!rise.array<11, scalar<f32>>) -> ()
            }) : () -> !rise.fun<array<9, scalar<f32>> -> array<11, scalar<f32>>>
            %4 = "rise.map"() {n = #rise.nat<9>, s = #rise.array<9, scalar<f32>>, t = #rise.array<11, scalar<f32>>} : () -> !rise.fun<fun<array<9, scalar<f32>> -> array<11, scalar<f32>>> -> fun<array<9, array<9, scalar<f32>>> -> array<9, array<11, scalar<f32>>>>>
            %5 = "rise.apply"(%4, %3, %0) : (!rise.fun<fun<array<9, scalar<f32>> -> array<11, scalar<f32>>> -> fun<array<9, array<9, scalar<f32>>> -> array<9, array<11, scalar<f32>>>>>, !rise.fun<array<9, scalar<f32>> -> array<11, scalar<f32>>>, !rise.array<9, array<9, scalar<f32>>>) -> !rise.array<9, array<11, scalar<f32>>>
            %6 = "rise.pad"() {l = #rise.nat<1>, n = #rise.nat<9>, r = #rise.nat<1>, t = #rise.array<11, scalar<f32>>} : () -> !rise.fun<array<9, array<11, scalar<f32>>> -> array<11, array<11, scalar<f32>>>>
            %7 = "rise.apply"(%6, %5) : (!rise.fun<array<9, array<11, scalar<f32>>> -> array<11, array<11, scalar<f32>>>>, !rise.array<9, array<11, scalar<f32>>>) -> !rise.array<11, array<11, scalar<f32>>>
            %8 = "rise.lambda"() ( {
            ^bb0(%arg4: !rise.array<11, scalar<f32>>):  // no predecessors
              %11 = "rise.slide"() {n = #rise.nat<9>, sp = #rise.nat<1>, sz = #rise.nat<3>, t = #rise.scalar<f32>} : () -> !rise.fun<array<11, scalar<f32>> -> array<9, array<3, scalar<f32>>>>
              %12 = "rise.apply"(%11, %arg4) : (!rise.fun<array<11, scalar<f32>> -> array<9, array<3, scalar<f32>>>>, !rise.array<11, scalar<f32>>) -> !rise.array<9, array<3, scalar<f32>>>
        %13 = "rise.lambda"() ( {
              ^bb0(%arg5: !rise.array<3, scalar<f32>>):  // no predecessors
                %16 = "rise.zip"() {n = #rise.nat<3>, s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<array<3, scalar<f32>> -> fun<array<3, scalar<f32>> -> array<3, tuple<scalar<f32>, scalar<f32>>>>>
                %17 = "rise.apply"(%16, %arg5, %1) : (!rise.fun<array<3, scalar<f32>> -> fun<array<3, scalar<f32>> -> array<3, tuple<scalar<f32>, scalar<f32>>>>>, !rise.array<3, scalar<f32>>, !rise.array<3, scalar<f32>>) -> !rise.array<3, tuple<scalar<f32>, scalar<f32>>>
                %18 = "rise.literal"() {literal = #rise.lit<0.000000, scalar<f32>>} : () -> !rise.scalar<f32>
                %19 = "rise.lambda"() ( {
                ^bb0(%arg6: !rise.tuple<scalar<f32>, scalar<f32>>, %arg7: !rise.scalar<f32>):  // no predecessors
                  %22 = "rise.fst"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
                  %23 = "rise.apply"(%22, %arg6) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
                  %24 = "rise.snd"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
                  %25 = "rise.apply"(%24, %arg6) : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
                  %26 = "rise.embed"(%23, %25, %arg7) ( {
                  ^bb0(%arg8: f32, %arg9: f32, %arg10: f32):  // no predecessors
                    %27 = mulf %arg8, %arg9 : f32
                    %28 = addf %arg10, %27 : f32
                    "rise.return"(%28) : (f32) -> ()
                  }) : (!rise.scalar<f32>, !rise.scalar<f32>, !rise.scalar<f32>) -> !rise.scalar<f32>
                  "rise.return"(%26) : (!rise.scalar<f32>) -> ()
                }) : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>
                %20 = "rise.reduceSeq"() {n = #rise.nat<3>, s = #rise.tuple<scalar<f32>, scalar<f32>>, t = #rise.scalar<f32>, to = "scf"} : () -> !rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<3, tuple<scalar<f32>, scalar<f32>>> -> scalar<f32>>>>
                %21 = "rise.apply"(%20, %19, %18, %17) : (!rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<3, tuple<scalar<f32>, scalar<f32>>> -> scalar<f32>>>>, !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>, !rise.scalar<f32>, !rise.array<3, tuple<scalar<f32>, scalar<f32>>>) -> !rise.scalar<f32>
                "rise.return"(%21) : (!rise.scalar<f32>) -> ()
              }) : () -> !rise.fun<array<3, scalar<f32>> -> scalar<f32>>
              %14 = "rise.mapSeq"() {n = #rise.nat<9>, s = #rise.array<3, scalar<f32>>, t = #rise.scalar<f32>, to = "scf"} : () -> !rise.fun<fun<array<3, scalar<f32>> -> scalar<f32>> -> fun<array<9, array<3, scalar<f32>>> -> array<9, scalar<f32>>>>
              %15 = "rise.apply"(%14, %13, %12) : (!rise.fun<fun<array<3, scalar<f32>> -> scalar<f32>> -> fun<array<9, array<3, scalar<f32>>> -> array<9, scalar<f32>>>>, !rise.fun<array<3, scalar<f32>> -> scalar<f32>>, !rise.array<9, array<3, scalar<f32>>>) -> !rise.array<9, scalar<f32>>
              "rise.return"(%15) : (!rise.array<9, scalar<f32>>) -> ()
            }) : () -> !rise.fun<array<11, scalar<f32>> -> array<9, scalar<f32>>>
            %9 = "rise.mapSeq"() {n = #rise.nat<11>, s = #rise.array<11, scalar<f32>>, t = #rise.array<9, scalar<f32>>, to = "scf"} : () -> !rise.fun<fun<array<11, scalar<f32>> -> array<9, scalar<f32>>> -> fun<array<11, array<11, scalar<f32>>> -> array<11, array<9, scalar<f32>>>>>
            %10 = "rise.apply"(%9, %8, %7) : (!rise.fun<fun<array<11, scalar<f32>> -> array<9, scalar<f32>>> -> fun<array<11, array<11, scalar<f32>>> -> array<11, array<9, scalar<f32>>>>>, !rise.fun<array<11, scalar<f32>> -> array<9, scalar<f32>>>, !rise.array<11, array<11, scalar<f32>>>) -> !rise.array<11, array<9, scalar<f32>>>
            "rise.out"(%arg3, %10) : (memref<9x9xf32>, !rise.array<11, array<9, scalar<f32>>>) -> ()
            "rise.return"() : () -> ()
          }) : () -> ()
          return
        }func @conv2DSeparable_test() {
          %c0 = constant 0 : index
          %c9 = constant 9 : index
          %c1 = constant 1 : index
          %c0_0 = constant 0 : index
          %c9_1 = constant 9 : index
          %c1_2 = constant 1 : index
          %c0_3 = constant 0 : index
          %c1_4 = constant 1 : index
          %c3 = constant 3 : index
          %c3_5 = constant 3 : index
          %0 = alloc() : memref<9x9xf32>
          %1 = alloc() : memref<3xf32>
          %2 = alloc() : memref<3xf32>
          %3 = alloc() : memref<9x9xf32>
          %cst = constant 0.000000e+00 : f32
          %cst_6 = constant 1.000000e+00 : f32
          %cst_7 = constant 2.000000e+00 : f32
          %4 = alloc() : memref<f32>
          store %cst_6, %4[] : memref<f32>
          scf.for %arg0 = %c0 to %c9 step %c1 {
            scf.for %arg1 = %c0_0 to %c9_1 step %c1_2 {
              %10 = load %4[] : memref<f32>
              store %10, %0[%arg0, %arg1] : memref<9x9xf32>
              %11 = load %4[] : memref<f32>
              %12 = addf %11, %cst_6 : f32
              store %12, %4[] : memref<f32>
            }
          }
          %5 = alloc() : memref<f32>
          store %cst_6, %5[] : memref<f32>
          scf.for %arg0 = %c0_3 to %c3 step %c1_4 {
            %10 = load %5[] : memref<f32>
            store %10, %1[%arg0] : memref<3xf32>
            %11 = load %5[] : memref<f32>
            store %11, %2[%arg0] : memref<3xf32>
          }
          call @conv2DSeparable(%0, %1, %2, %3) : (memref<9x9xf32>, memref<3xf32>, memref<3xf32>, memref<9x9xf32>) -> ()
          %6 = memref_cast %0 : memref<9x9xf32> to memref<*xf32>
          call @print_memref_f32(%6) : (memref<*xf32>) -> ()
          %7 = memref_cast %1 : memref<3xf32> to memref<*xf32>
          call @print_memref_f32(%7) : (memref<*xf32>) -> ()
          %8 = memref_cast %2 : memref<3xf32> to memref<*xf32>
          call @print_memref_f32(%8) : (memref<*xf32>) -> ()
          %9 = memref_cast %3 : memref<9x9xf32> to memref<*xf32>
          call @print_memref_f32(%9) : (memref<*xf32>) -> ()
          return
        }