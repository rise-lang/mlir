rise.module {
    //Array
    %array0 = rise.literal #rise.lit<array<4, !rise.int, [1,2,3,4]>>


    //Reduction
    %addFun = rise.add #rise.int
    %initializer = rise.literal #rise.lit<int<4>>
    %reduce4Ints = rise.reduce #rise.nat<4> #rise.int #rise.int
    %result = rise.apply %reduce4Ints, %addFun, %initializer, %array0

    rise.return %result : !rise.data<int>
}