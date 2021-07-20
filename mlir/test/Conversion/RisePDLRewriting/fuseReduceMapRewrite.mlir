module @pdl_patterns {

//    // *g >> reduce f init -> reduce (acc, x => f acc (g x)) init
//    @rule def reduceMapFusion: Strategy[Rise] = {
//      case e @ App(App(App(r @ ReduceX(), f), init), App(App(map(), g), in)) =>
//        val red = (r, g.t) match {
//          case (reduce(), FunType(i, o)) if i =~= o => reduce
//          case _ => reduceSeq
//        }
//        Success(red(fun(acc => fun(x =>
//          preserveType(f)(acc)(preserveType(g)(x)))))(init)(in) !: e.t)
//    }

  pdl.pattern @fuseReduceMap : benefit(2) {
    %n = pdl.attribute
    %s = pdl.attribute
    %t = pdl.attribute
    %mapResT = pdl.type
    %applyMapResT = pdl.type
    %gT = pdl.type
    %g = pdl.operation "rise.lambda" -> (%gT : !pdl.type)                // !rise.fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>>
    %gRes = pdl.result 0 of %g
    %array = pdl.operand
    %map = pdl.operation "rise.mapSeq" {"n" = %n, "s" = %s, "t" = %t} -> (%mapResT : !pdl.type)
    %mapRes = pdl.result 0 of %map
    %mapApply = pdl.operation "rise.apply"(%mapRes, %gRes, %array : !pdl.value, !pdl.value, !pdl.value) -> (%applyMapResT : !pdl.type)
    %mapApplyRes = pdl.result 0 of %mapApply

    %t2 = pdl.attribute
    %fT = pdl.type
    %reduceResT = pdl.type
    %applyReduceResT = pdl.type
    %f = pdl.operation "rise.lambda" -> (%fT : !pdl.type)                // !rise.fun<tuple<scalar<f32>, scalar<f32>> -> fun<scalar<f32> -> scalar<f32>>>
    %fRes = pdl.result 0 of %f
    %init = pdl.operand
    %reduce = pdl.operation "rise.reduceSeq" {"t" = %t2} -> (%reduceResT : !pdl.type)
    %reduceRes = pdl.result 0 of %reduce
    %redApply = pdl.operation "rise.apply"(%reduceRes, %fRes, %init, %mapApplyRes : !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%applyReduceResT : !pdl.type)
    %redApplyRes = pdl.result 0 of %redApply

    pdl.rewrite %redApply {
      %sT = pdl.apply_native_rewrite "typeFromAttr"(%s : !pdl.attribute) : !pdl.type
      %tT = pdl.apply_native_rewrite "typeFromAttr"(%t : !pdl.attribute) : !pdl.type
      %t2T = pdl.apply_native_rewrite "typeFromAttr"(%t2 : !pdl.attribute) : !pdl.type

      %reduceLambdaT_0 = pdl.apply_native_rewrite "rise.fun"(%sT, %t2T : !pdl.type, !pdl.type) : !pdl.type
      %reduceLambdaT = pdl.apply_native_rewrite "rise.fun"(%reduceLambdaT_0, %t2T : !pdl.type, !pdl.type) : !pdl.type     // reduceLambdaT: fun<s -> fun<t2 -> t2>>
      %reduceLambdaOp = pdl.operation "rise.tmp" -> (%reduceLambdaT : !pdl.type)

      // To enable parsing of this rewrite comment out the lambda and its return below and remove the following comments
//      %reduceLambda = pdl.result 0 of %reduceLambdaOp
//      %xOp = pdl.operation "rise.tmp" -> (%xT : !pdl.type)
//      %x = pdl.result 0 of %xOp
//      %accOp = pdl.operation "rise.tmp" -> (%accT : !pdl.type)
//      %acc = pdl.result 0 of %accOp

      %xT = pdl.apply_native_rewrite "typeFromAttr"(%s : !pdl.attribute) : !pdl.type
      %accT = pdl.apply_native_rewrite "typeFromAttr"(%t2 : !pdl.attribute) : !pdl.type

      %reduceLambda = pdl.operation "rise.lambda" %x, %acc : %xT, %accT {                                       // other notation chosen to indicate BlockArgs rather than operands
        // Problem:
        // The applications of the two lambdas will not have the same parent region like the lambda.
        // For subsequent rewrites it would be better to move (clone) the lambdas into this region.
        // Currently we do not have options for this in PDL
        %applygT_0 = pdl.apply_native_rewrite "fun_output"(%gT : !pdl.type) : !pdl.type
        %applygT = pdl.apply_native_rewrite "fun_output"(%applygT_0 : !pdl.type) : !pdl.type
        %applyg = pdl.operation "rise.apply"(%gRes, %x : !pdl.value, !pdl.value) -> (%applygT : !pdl.type)
        %applygRes = pdl.result 0 of %applyg

        %applyfT_0 = pdl.apply_native_rewrite "fun_output"(%fT : !pdl.type) : !pdl.type
        %applyfT = pdl.apply_native_rewrite "fun_output"(%applyfT_0 : !pdl.type) : !pdl.type
        %applyf = pdl.operation "rise.apply"(%fRes, %applygRes, %acc : !pdl.value, !pdl.value, !pdl.value) -> (%applyfT : !pdl.type)
        %applyfRes = pdl.result 0 of %applyf

        %return = pdl.operation "rise.return"(%applyfRes : !pdl.value)
      }

      // building funType from back:
      %arrayT = pdl.apply_native_rewrite "rise.array"(%n, %sT : !pdl.attribute, !pdl.type) : !pdl.type // can't get the type from %array above bcs we don't want to restrict the op that creates that value
      %newApplyReduceResT = pdl.apply_native_rewrite "rise.array"(%n, %t2T : !pdl.attribute, !pdl.type) : !pdl.type
      %reduceT_0 = pdl.apply_native_rewrite "rise.fun"(%arrayT, %newApplyReduceResT : !pdl.type, !pdl.type) : !pdl.type
      %reduceT_1 = pdl.apply_native_rewrite "rise.fun"(%t2T, %reduceT_0 : !pdl.type, !pdl.type) : !pdl.type
      %reduceT = pdl.apply_native_rewrite "rise.fun"(%reduceLambdaT, %reduceT_1 : !pdl.type, !pdl.type) : !pdl.type

      %newReduce = pdl.operation "rise.reduceSeq"{"n" = %n, "s" = %t, "t" = %t2} -> (%reduceT : !pdl.type)
      %newReduceRes = pdl.result 0 of %newReduce
      %newReduceApply = pdl.operation "rise.apply"(%newReduceRes, %reduceLambda, %init, %array : !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%newApplyReduceResT : !pdl.type)

      // cleanup
      pdl.erase %mapApply
      pdl.erase %map
      pdl.replace %redApply with %newReduceApply
      pdl.erase %reduce
    }
  }
}


module @ir {
func @rise_fun(%arg0: memref<6x6xf32>, %arg1: memref<6x6xf32>, %arg2: memref<6x6xf32>) {
  "rise.lowering_unit"() ( {
    %0 = "rise.in"(%arg1) {"ksc.color" = [0.5 : f32, 0.5 : f32, 0.5 : f32]} : (memref<6x6xf32>) -> !rise.array<6, array<6, scalar<f32>>>
    %1 = "rise.in"(%arg2) {"ksc.color" = [0.5 : f32, 0.5 : f32, 0.5 : f32]} : (memref<6x6xf32>) -> !rise.array<6, array<6, scalar<f32>>>
    %2 = "rise.transpose"() {m = #rise.nat<6>, n = #rise.nat<6>, t = #rise.scalar<f32>} : () -> !rise.fun<array<6, array<6, scalar<f32>>> -> array<6, array<6, scalar<f32>>>>
    %3 = "rise.apply"(%2, %1) {"ksc.color" = [0.5 : f32, 0.5 : f32, 0.5 : f32]} : (!rise.fun<array<6, array<6, scalar<f32>>> -> array<6, array<6, scalar<f32>>>>, !rise.array<6, array<6, scalar<f32>>>) -> !rise.array<6, array<6, scalar<f32>>>
    %4 = "rise.lambda"() ( {
    ^bb0(%arg3: !rise.array<6, scalar<f32>>):  // no predecessors
      %7 = "rise.lambda"() ( {
      ^bb0(%arg4: !rise.array<6, scalar<f32>>):  // no predecessors
        %10 = "rise.zip"() {n = #rise.nat<6>, s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<array<6, scalar<f32>> -> fun<array<6, scalar<f32>> -> array<6, tuple<scalar<f32>, scalar<f32>>>>>
        %11 = "rise.apply"(%10, %arg3, %arg4) {"ksc.color" = [0.6 : f32, 0.1 : f32, 0.8 : f32]} : (!rise.fun<array<6, scalar<f32>> -> fun<array<6, scalar<f32>> -> array<6, tuple<scalar<f32>, scalar<f32>>>>>, !rise.array<6, scalar<f32>>, !rise.array<6, scalar<f32>>) -> !rise.array<6, tuple<scalar<f32>, scalar<f32>>>
        %12 = "rise.lambda"() ( {
        ^bb0(%arg5: !rise.tuple<scalar<f32>, scalar<f32>>):  // no predecessors
          %19 = "rise.fst"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
          %20 = "rise.snd"() {s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
          %21 = "rise.apply"(%19, %arg5) {"ksc.color" = [0.9 : f32, 0.6 : f32, 0.2 : f32]} : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
          %22 = "rise.apply"(%20, %arg5) {"ksc.color" = [0.5 : f32, 0.1 : f32, 0.9 : f32]} : (!rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.tuple<scalar<f32>, scalar<f32>>) -> !rise.scalar<f32>
          %23 = "rise.embed"(%21, %22) ( {
          ^bb0(%arg6: f32, %arg7: f32):  // no predecessors
            %24 = mulf %arg6, %arg7 {"ksc.color" = [0.5 : f32, 0.5 : f32, 0.5 : f32]} : f32
            "rise.return"(%24) : (f32) -> ()
          }) {"ksc.color" = [0.5 : f32, 0.5 : f32, 0.5 : f32]} : (!rise.scalar<f32>, !rise.scalar<f32>) -> !rise.scalar<f32>
          "rise.return"(%23) : (!rise.scalar<f32>) -> ()
        }) : () -> !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>
        %13 = "rise.mapSeq"() {n = #rise.nat<6>, s = #rise.tuple<scalar<f32>, scalar<f32>>, t = #rise.scalar<f32>} : () -> !rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>> -> fun<array<6, tuple<scalar<f32>, scalar<f32>>> -> array<6, scalar<f32>>>>
        %14 = "rise.apply"(%13, %12, %11) {"ksc.color" = [0.9 : f32, 0.9 : f32, 0.0 : f32]} : (!rise.fun<fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>> -> fun<array<6, tuple<scalar<f32>, scalar<f32>>> -> array<6, scalar<f32>>>>, !rise.fun<tuple<scalar<f32>, scalar<f32>> -> scalar<f32>>, !rise.array<6, tuple<scalar<f32>, scalar<f32>>>) -> !rise.array<6, scalar<f32>>
        %15 = "rise.lambda"() ( {
        ^bb0(%arg5: !rise.scalar<f32>, %arg6: !rise.scalar<f32>):  // no predecessors
          %19 = "rise.embed"(%arg5, %arg6) ( {
          ^bb0(%arg7: f32, %arg8: f32):  // no predecessors
            %20 = addf %arg7, %arg8 {"ksc.color" = [0.5 : f32, 0.5 : f32, 0.5 : f32]} : f32
            "rise.return"(%20) : (f32) -> ()
          }) {"ksc.color" = [0.5 : f32, 0.5 : f32, 0.5 : f32]} : (!rise.scalar<f32>, !rise.scalar<f32>) -> !rise.scalar<f32>
          "rise.return"(%19) : (!rise.scalar<f32>) -> ()
        }) : () -> !rise.fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>>
        %16 = "rise.literal"() {literal = #rise.lit<0.000000, scalar<f32>>, "ksc.color" = [0.6 : f32, 0.6 : f32, 0.1 : f32]} : () -> !rise.scalar<f32>
        %17 = "rise.reduceSeq"() {n = #rise.nat<6>, s = #rise.scalar<f32>, t = #rise.scalar<f32>} : () -> !rise.fun<fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<6, scalar<f32>> -> scalar<f32>>>>
        %18 = "rise.apply"(%17, %15, %16, %14) {"ksc.color" = [0.6 : f32, 0.6 : f32, 0.1 : f32]} : (!rise.fun<fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>> -> fun<scalar<f32> -> fun<array<6, scalar<f32>> -> scalar<f32>>>>, !rise.fun<scalar<f32> -> fun<scalar<f32> -> scalar<f32>>>, !rise.scalar<f32>, !rise.array<6, scalar<f32>>) -> !rise.scalar<f32>
        "rise.return"(%18) : (!rise.scalar<f32>) -> ()
      }) : () -> !rise.fun<array<6, scalar<f32>> -> scalar<f32>>
      %8 = "rise.mapSeq"() {n = #rise.nat<6>, s = #rise.array<6, scalar<f32>>, t = #rise.scalar<f32>} : () -> !rise.fun<fun<array<6, scalar<f32>> -> scalar<f32>> -> fun<array<6, array<6, scalar<f32>>> -> array<6, scalar<f32>>>>
      %9 = "rise.apply"(%8, %7, %3) {"ksc.color" = [0.8 : f32, 0.6 : f32, 0.5 : f32]} : (!rise.fun<fun<array<6, scalar<f32>> -> scalar<f32>> -> fun<array<6, array<6, scalar<f32>>> -> array<6, scalar<f32>>>>, !rise.fun<array<6, scalar<f32>> -> scalar<f32>>, !rise.array<6, array<6, scalar<f32>>>) -> !rise.array<6, scalar<f32>>
      "rise.return"(%9) : (!rise.array<6, scalar<f32>>) -> ()
    }) : () -> !rise.fun<array<6, scalar<f32>> -> array<6, scalar<f32>>>
    %5 = "rise.mapSeq"() {n = #rise.nat<6>, s = #rise.array<6, scalar<f32>>, t = #rise.array<6, scalar<f32>>} : () -> !rise.fun<fun<array<6, scalar<f32>> -> array<6, scalar<f32>>> -> fun<array<6, array<6, scalar<f32>>> -> array<6, array<6, scalar<f32>>>>>
    %6 = "rise.apply"(%5, %4, %0) {"ksc.color" = [0.9 : f32, 0.9 : f32, 0.0 : f32]} : (!rise.fun<fun<array<6, scalar<f32>> -> array<6, scalar<f32>>> -> fun<array<6, array<6, scalar<f32>>> -> array<6, array<6, scalar<f32>>>>>, !rise.fun<array<6, scalar<f32>> -> array<6, scalar<f32>>>, !rise.array<6, array<6, scalar<f32>>>) -> !rise.array<6, array<6, scalar<f32>>>
    "rise.out"(%arg0, %6) : (memref<6x6xf32>, !rise.array<6, array<6, scalar<f32>>>) -> ()
    "rise.return"() : () -> ()
  }) : () -> ()
  return
}
}