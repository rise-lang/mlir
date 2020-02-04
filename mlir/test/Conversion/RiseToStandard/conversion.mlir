module {
%someInt = rise.literal #rise.lit<int<42>>
%someFloat = rise.literal #rise.lit<float<42>>
%someOtherFloat = rise.literal #rise.lit<float<13>>


%id = rise.lambda (%i) : !rise.fun<data<int> -> data<int>> {
    rise.return %i : !rise.data<int>
}



}