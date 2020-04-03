// RUN: mlir-opt %s

module {
func @rise_dot_product_simplified() {
    //Arrays
    %array0 = rise.literal #rise.lit<array<10, !rise.int, [1,2,3,4,5,6,7,8,9,10]>>
    %array1 = rise.literal #rise.lit<array<10, !rise.int, [1,2,3,4,5,6,7,8,9,10]>>

    //Zipping
    %zipFun = rise.zip #rise.nat<10> #rise.int #rise.int
    %zippedArrays = rise.apply %zipFun : !rise.fun<data<array<10, int>> -> fun<data<array<10, int>> -> data<array<10, tuple<int, int>>>>>, %array0, %array1

    //Multiply

    %tupleMultFun = rise.lambda (%tuple) : !rise.fun<data<tuple<int, int>> -> data<int>> {
        %fstFun = rise.fst #rise.int #rise.int
        %sndFun = rise.snd #rise.int #rise.int

        %fst = rise.apply %fstFun : !rise.fun<data<tuple<int, int>> -> data<int>> ,%tuple
        %snd = rise.apply %sndFun : !rise.fun<data<tuple<int, int>> -> data<int>> ,%tuple

        %multFun = rise.mult #rise.int
        %result = rise.apply %multFun : !rise.fun<data<int> -> fun<data<int> -> data<int>>>, %snd, %fst

        rise.return %result : !rise.data<int>
        //return %result : !rise.data<int>
    }
    %map10TuplesToInts = rise.map #rise.nat<10> #rise.tuple<int, int> #rise.int
    %multipliedArray = rise.apply %map10TuplesToInts : !rise.fun<fun<data<tuple<int, int>> -> data<int>> -> fun<data<array<10, tuple<int, int>>> -> data<array<10, int>>>>, %tupleMultFun, %zippedArrays

    //Reduction
    %addFun = rise.add #rise.int
    %initializer = rise.literal #rise.lit<int<0>>
    %reduce10Ints = rise.reduce #rise.nat<10> #rise.int #rise.int
    %result = rise.apply %reduce10Ints : !rise.fun<fun<data<int> -> fun<data<int> -> data<int>>> -> fun<data<int> -> fun<data<array<10, int>> -> data<int>>>>, %addFun, %initializer, %multipliedArray

    rise.return %result : !rise.data<int>
}
}
