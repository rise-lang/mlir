// RUN: mlir-opt %s -convert-rise-to-imperative -convert-linalg-to-loops -convert-loop-to-std -convert-std-to-llvm | mlir-cpu-runner -e array_times_2 -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext  | FileCheck %s --check-prefix=ARRAY_TIMES_2

func @print_memref_f32(memref<*xf32>)
func @rise_fun(memref<4xf32>)
func @array_times_2() {

    rise.fun "rise_fun" (%outArg:memref<4xf32>) {
        %out = rise.out %outArg
        %array = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>
        %doubleFun = rise.lambda (%summand) : !rise.fun<data<float> -> data<float>> {
            %addFun = rise.add #rise.float
            %doubled = rise.apply %addFun, %summand, %summand
            rise.return %doubled : !rise.data<float>
        }
        %map4IntsToInts = rise.map #rise.nat<4> #rise.float #rise.float
        %doubledArray = rise.apply %map4IntsToInts, %doubleFun, %array

        rise.return %doubledArray : !rise.data<array<4, float>>
    }

    //prepare output Array
    %outputArray = alloc() : memref<4xf32>
    call @rise_fun(%outputArray) : (memref<4xf32>) -> ()

    %print_me = memref_cast %outputArray : memref<4xf32> to memref<*xf32>
    call @print_memref_f32(%print_me): (memref<*xf32>) -> ()
    return
}
// ARRAY_TIMES_2: Unranked Memref rank = 1 descriptor@ = {{.*}}
// ARRAY_TIMES_2: Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// ARRAY_TIMES_2: [10, 10, 10, 10]



Michel Steuwer:
con(array)(λ(expT(n`.`dt1, read))(x =>
        comment("mapSeq")`;`
        `for`(n, i =>
//          λ(expT(dt1, read))(x => λ(accT(dt2))(o => acc(f(x))(o)))(x `@` i)(A `@` i)
//          λ(accT(dt2))(o => acc(f(x `@` i))(o))(A `@` i)
            acc(f(x `@` i))(A `@` i)
        )
    ))
Michel Steuwer:
con(array)(λ(expT(n`.`dt1, read))(x =>
        comment("mapSeq")`;`
        `for`(n, i => {
          //          λ(expT(dt1, read))(x => λ(accT(dt2))(o => acc(f(x))(o)))(x `@` i)(A `@` i)
          //          λ(accT(dt2))(o => acc(f(x `@` i))(o))(A `@` i)
          val `x@i` = x `@` i     // %xi = rise.idx %x %i
          val `fx@i` = f(`x@i`)   // %fxi = rise.apply f %xi
          val `A@i` = A `@` i     // %outi = rise.idx %out %i
          acc(`fx@i`)(`A@i`)
        })
    ))
Michel Steuwer:
con(array)(λ(expT(n`.`dt1, read))(x =>
        comment("mapSeq")`;`
        `for`(n, i => {
          // %xi = rise.idx %x %i
          // %outi = rise.idx %out %i
          // %expr = unary_op %xi
          // rise.assign %expr %outi
        })
    ))