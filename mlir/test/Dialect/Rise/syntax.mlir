// RUN: mlir-opt %s

module {
    func @rise_id() {
    ^id:
        //integer literal
        %42 = rise.literal #rise.lit<int<42>>

        %id = rise.lambda (%i) : !rise.fun<data<int> -> data<int>> {
            rise.return %i : !rise.data<int>
        }
        %result = rise.apply %id : !rise.fun<int -> int>, %42

        "rise.return"() : () -> ()
    }

    func @rise_add_example() {
        %int0 = rise.literal #rise.lit<int<7>>
        %int1 = rise.literal #rise.lit<int<13>>

        %addFun = rise.lambda (%summand0) : !rise.fun<data<int> -> fun<data<int> -> data<int>>> {
            %addWithSummand0 = rise.lambda (%summand1) : !rise.fun<data<int> -> data<int>> {
                %addFun = rise.add #rise.int
                %addition = rise.apply %addFun : !rise.fun<data<int> -> fun<data<int> -> data<int>>>, %summand0, %summand1
                rise.return %addition : !rise.data<int>
            }
            rise.return %addWithSummand0 : !rise.fun<int -> int>
        }
        %addWithInt0 = rise.apply %addFun : !rise.fun<int -> !rise.fun<int -> int>>, %int0
        %result = rise.apply %addWithInt0 : !rise.fun<int -> int>, %int1

        "rise.return"() : () -> ()
    }

    func @rise_sub_example() {
        %int0 = rise.literal #rise.lit<int<7>>
        %int1 = rise.literal #rise.lit<int<13>>

        %subFun = rise.lambda (%op0) : !rise.fun<data<int> -> fun<data<int> -> data<int>>> {
            %subWithOp0 = rise.lambda (%op1) : !rise.fun<data<int> -> data<int>> {
                %subFun = rise.sub #rise.int
                %difference = rise.apply %subFun : !rise.fun<data<int> -> fun<data<int> -> data<int>>>, %op0, %op1
                rise.return %difference : !rise.data<int>
            }
            rise.return %subWithOp0 : !rise.fun<int -> int>
        }
        %subWithInt0 = rise.apply %subFun : !rise.fun<int -> !rise.fun<int -> int>>, %int0
        %result = rise.apply %subWithInt0 : !rise.fun<int -> int>, %int1

        "rise.return"() : () -> ()
    }

    func @rise_mult_example() {
        %int0 = rise.literal #rise.lit<int<7>>
        %int1 = rise.literal #rise.lit<int<13>>

        %multFun = rise.lambda (%op0) : !rise.fun<data<int> -> fun<data<int> -> data<int>>> {
            %multWithOp0 = rise.lambda (%op1) : !rise.fun<data<int> -> data<int>> {
                %multFun = rise.mult #rise.int
                %product = rise.apply %multFun : !rise.fun<data<int> -> fun<data<int> -> data<int>>>, %op0, %op1
                rise.return %product : !rise.data<int>
            }
            rise.return %multWithOp0 : !rise.fun<int -> int>
        }
        %multWithInt0 = rise.apply %multFun : !rise.fun<int -> !rise.fun<int -> int>>, %int0
        %result = rise.apply %multWithInt0 : !rise.fun<int -> int>, %int1

        "rise.return"() : () -> ()
    }

    func @rise_div_example() {
        %int0 = rise.literal #rise.lit<int<7>>
        %int1 = rise.literal #rise.lit<int<13>>

        %divFun = rise.lambda (%op0) : !rise.fun<data<int> -> fun<data<int> -> data<int>>> {
            %divWithOp0 = rise.lambda (%op1) : !rise.fun<data<int> -> data<int>> {
                %divFun = rise.div #rise.int
                %quotient = rise.apply %divFun : !rise.fun<data<int> -> fun<data<int> -> data<int>>>, %op0, %op1
                rise.return %quotient : !rise.data<int>
            }
            rise.return %divWithOp0 : !rise.fun<int -> int>
        }
        %divWithInt0 = rise.apply %divFun : !rise.fun<int -> !rise.fun<int -> int>>, %int0
        %result = rise.apply %divWithInt0 : !rise.fun<int -> int>, %int1

        "rise.return"() : () -> ()
    }

    func @rise_tuple_example() {
        //creating a simple tuple of an int and a float
        %int0 = rise.literal #rise.lit<int<7>>
        %float0 = rise.literal #rise.lit<float<13>>

        %tupleFun = rise.tuple #rise.int #rise.float
        %tupleWithInt0 = rise.apply %tupleFun : !rise.fun<data<int> -> fun<data<float> -> data<tuple<int, float>>>>, %int0
        %tupleIntFloat = rise.apply %tupleWithInt0 : !rise.fun<data<float> -> data<tuple<int, float>>>, %float0

        "rise.return"() : () -> ()
    }

    func @rise_zip_example() {
        //zipping two arrays
        %array0 = rise.literal #rise.lit<array<2, !rise.int, [1,2]>>
        %array1 = rise.literal #rise.lit<array<2, !rise.int, [1,2]>>

        %zipFun = rise.zip #rise.nat<2> #rise.int #rise.int
        %zipWithArray0 = rise.apply %zipFun : !rise.fun<data<array<2, int>> -> fun<data<array<2, int>> -> data<array<2, tuple<int, int>>>>>, %array0
        %zippedArrays  = rise.apply %zipWithArray0 : !rise.fun<data<array<2, int>> -> data<array<2, tuple<int, int>>>>, %array1

        "rise.return"() : () -> ()
    }

    func @rise_map_example() {
        %array = rise.literal #rise.lit<array<10, !rise.int, [1,2,3,4,5,6,7,8,9,10]>>
        %doubleFun = rise.lambda (%summand) : !rise.fun<data<int> -> data<int>> {
            %addFun = rise.add #rise.int
            %double = rise.apply %addFun : !rise.fun<data<int> -> fun<data<int> -> data<int>>>, %summand, %summand
            rise.return %double : !rise.data<int>
        }
        %map10IntsToInts = rise.map #rise.nat<10> #rise.int #rise.int
        %mapDoubleFun = rise.apply %map10IntsToInts : !rise.fun<fun<data<int> -> data<int>> -> fun<data<array<10, int>> -> data<array<10, int>>>>, %doubleFun
        %doubledArray = rise.apply %mapDoubleFun : !rise.fun<data<array<10, int>> -> data<array<10, int>>>, %array

        "rise.return"() : () -> ()
    }

///proposed structure:
//        // Natural numbers: N =
//        !rise.nat<2>
//
//        // Data types: DT =
//        !rise.int
//        !rise.Array<N, DT>
//        !rise.Tuple<DT, DT>
//        !rise.natAsData<N>
//
//        // Rise Types: T =
//        !rise.fun<T, T>
//        !rise.data<DT>

//         // FUTURE with dependent functions (+ their types)
//         %z0 = rise.zip : (N: !rise.nat<2>) -> !rise.Array<N, > -> !rise.Array<N, > -> !rise.Array<N, >
//         %z1 = rise.depApply %z0 !rise.nat<2>
//         %z2 = rise.depApply %z1 !rise.int
//         %z3 = rise.depApply %z2 !rise.int
//         %z4 = rise.apply    %z3 %array0
//         %zz = rise.apply    %z4 %array1
}
