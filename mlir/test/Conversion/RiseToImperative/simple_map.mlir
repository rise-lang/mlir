func @simple_map_example() {

    res = rise.fun {
        %array0 = rise.literal #rise.lit<array<4, !rise.float, [5,5,5,5]>>
        %doubleFun = rise.lambda (%summand) : !rise.fun<data<int> -> data<int>> {
            %addFun = rise.add #rise.int
            %double = rise.apply %addFun : !rise.fun<data<int> -> fun<data<int> -> data<int>>>, %summand, %summand
            rise.return %double : !rise.data<int>
        }
        %map10IntsToInts = rise.map #rise.nat<10> #rise.int #rise.int
        %mapDoubleFun = rise.apply %map10IntsToInts : !rise.fun<fun<data<int> -> data<int>> -> fun<data<array<10, int>> -> data<array<10, int>>>>, %doubleFun
        %doubledArray = rise.apply %mapDoubleFun : !rise.fun<data<array<10, int>> -> data<array<10, int>>>, %array

        rise.return %doubledArray
    }

}