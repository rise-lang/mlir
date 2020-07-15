// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-std -lower-affine -convert-scf-to-std -convert-std-to-llvm | mlir-cpu-runner -e map_slide_reduce -entry-point-result=void -O3 -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=MAP_SLIDE_REDUCE

func @print_memref_f32(memref<*xf32>)
func @rise_fun(%outArg:memref<9x7xf32>, %inA:memref<9x9xf32>) {

    %A = rise.in %inA : !rise.array<9, array<9, scalar<f32>>>


    %slideLambda = rise.lambda (%arr : !rise.array<9, scalar<f32>>) -> !rise.array<7, array<3, scalar<f32>>> {
        %slide = rise.slide #rise.nat<7> #rise.nat<3> #rise.nat<1> #rise.scalar<f32> // what does n here mean?
        %slizzled = rise.apply %slide, %arr
        rise.return %slizzled : !rise.array<7, array<3, scalar<f32>>>
    }
    %map = rise.map #rise.nat<9> #rise.array<9, scalar<f32>> #rise.array<7, array<3, scalar<f32>>>
    %slidemapped = rise.apply %map, %slideLambda, %A

    %mapLambda = rise.lambda (%array : !rise.array<7, array<3, scalar<f32>>>) -> !rise.array<7, scalar<f32>> {
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
        %mapReduce = rise.mapSeq {to = "affine"}  #rise.nat<7> #rise.array<3, scalar<f32>> #rise.scalar<f32>
        %result = rise.apply %mapReduce, %reduceWindow, %array
        rise.return %result : !rise.array<7, scalar<f32>>
    }
    %mapOuter = rise.mapSeq {to = "affine"}  #rise.nat<9> #rise.array<7, array<3, scalar<f32>>> #rise.array<7, scalar<f32>>
    %result = rise.apply %mapOuter, %mapLambda, %slidemapped
    rise.out %outArg <- %result
    return
}
func @rtclock() -> (f64)
func @print_flops(f64,f64,i64)
func @map_slide_reduce() {
    //prepare output Array
    %outputArray = alloc() : memref<9x7xf32>

    %A = alloc() : memref<9x9xf32>

    %cst0 = constant 0.000000e+00 : f32
    %memrefcst1 = alloc() : memref<f32>
    %cst1 = constant 1.000000e+00 : f32
    store %cst1, %memrefcst1[] : memref<f32>

    %val = alloc() : memref<f32>
    store %cst0, %val[] : memref<f32>

    %c0 = constant 0 : index
    %c16 = constant 9 : index
    %c1 = constant 1 : index
    scf.for %arg0 = %c0 to %c16 step %c1 {
        scf.for %arg1 = %c0 to %c16 step %c1 {
            %val_loaded = load %val[] : memref<f32>
            %cst1_loaded = load %memrefcst1[] : memref<f32>
            %interm = addf %val_loaded, %cst1_loaded : f32
            store %interm, %val[] : memref<f32>
            store %interm, %A[%arg1, %arg0] : memref<9x9xf32>
        }
    }

    call @rise_fun(%outputArray, %A) : (memref<9x7xf32>, memref<9x9xf32>) -> ()

    %print_meA = memref_cast %A : memref<9x9xf32> to memref<*xf32>
    call @print_memref_f32(%print_meA): (memref<*xf32>) -> ()
    %print_me = memref_cast %outputArray : memref<9x7xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// input is:
//[[1,   10,   19,   28,   37,   46,   55,   64,   73],
// [2,   11,   20,   29,   38,   47,   56,   65,   74],
// [3,   12,   21,   30,   39,   48,   57,   66,   75],
// [4,   13,   22,   31,   40,   49,   58,   67,   76],
// [5,   14,   23,   32,   41,   50,   59,   68,   77],
// [6,   15,   24,   33,   42,   51,   60,   69,   78],
// [7,   16,   25,   34,   43,   52,   61,   70,   79],
// [8,   17,   26,   35,   44,   53,   62,   71,   80],
// [9,   18,   27,   36,   45,   54,   63,   72,   81]]
//output:
// MAP_SLIDE_REDUCE: Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [9, 9] strides = [9, 1] data =
// MAP_SLIDE_REDUCE: {{[[30,   57,   84,   111,   138,   165,   192],}}
// MAP_SLIDE_REDUCE:  [33,   60,   87,   114,   141,   168,   195],
// MAP_SLIDE_REDUCE:  [36,   63,   90,   117,   144,   171,   198],
// MAP_SLIDE_REDUCE:  [39,   66,   93,   120,   147,   174,   201],
// MAP_SLIDE_REDUCE:  [42,   69,   96,   123,   150,   177,   204],
// MAP_SLIDE_REDUCE:  [45,   72,   99,   126,   153,   180,   207],
// MAP_SLIDE_REDUCE:  [48,   75,   102,   129,   156,   183,   210],
// MAP_SLIDE_REDUCE:  [51,   78,   105,   132,   159,   186,   213],
// MAP_SLIDE_REDUCE:  {{[54,   81,   108,   135,   162,   189,   216]]}}