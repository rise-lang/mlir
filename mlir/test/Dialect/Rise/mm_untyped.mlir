    // Matrices
    %A = rise.literal #rise.lit<array<4.4, !rise.int, [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]>>
    %B = rise.literal #rise.lit<array<4.4, !rise.int, [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]>>

    %m1fun = rise.lambda (%arow) : !rise.fun<data<array<4, int>> -> data<array<4, int>>> {
        %m2fun = rise.lambda (%bcol) : !rise.fun<data<array<4, int>> -> data<array<4, int>>> {

            //Zipping
            %zipFun = rise.zip #rise.nat<4> #rise.int #rise.int
            %zippedArrays = rise.apply %zipFun, %arow, %bcol

            //Multiply
            %tupleMultFun = rise.lambda (%tuple) : !rise.fun<data<tuple<int, int>> -> data<int>> {
                %fstFun = rise.fst #rise.int #rise.int
                %sndFun = rise.snd #rise.int #rise.int

                %fst = rise.apply %fstFun, %tuple
                %snd = rise.apply %sndFun, %tuple

                %multFun = rise.mult #rise.int
                %result = rise.apply %multFun, %snd, %fst

                rise.return %result : !rise.data<int>
            }
            %map10TuplesToInts = rise.map #rise.nat<4> #rise.tuple<int, int> #rise.int
            %multipliedArray = rise.apply %map10TuplesToInts, %tupleMultFun, %zippedArrays

            //Reduction
            %addFun = rise.add #rise.int
            %initializer = rise.literal #rise.lit<int<0>>
            %reduce10Ints = rise.reduce #rise.nat<4> #rise.int #rise.int
            %result = rise.apply %reduce10Ints, %addFun, %initializer, %multipliedArray

            rise.return %result : !rise.data<int>
        }
        %m2 = rise.map #rise.nat<4> #rise.array<4, int> #rise.array<4, int>
        %result = rise.apply %m2, %m2fun, %B
        rise.return %result : !rise.data<array<4, array<4, int>>>
    }
    %m1 = rise.map #rise.nat<4> #rise.array<4, !rise.int> #rise.array<4, !rise.int>
    %result = rise.apply %m1, %m1fun, %A
