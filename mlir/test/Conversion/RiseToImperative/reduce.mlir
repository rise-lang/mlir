rise.module { // to be rise interface
    //Array
    %array0 = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>

    //Reduction
    %addFun = rise.add #rise.float
    %initializer = rise.literal #rise.lit<float<0>>
    %reduce4Ints = rise.reduce #rise.nat<4> #rise.float #rise.float
    %result = rise.apply %reduce4Ints, %addFun, %initializer, %array0

    rise.return %result : !rise.data<float>
}