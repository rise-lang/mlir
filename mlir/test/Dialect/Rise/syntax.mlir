// RUN: mlir-opt -split-input-file %s | mlir-opt

// note: see actual "rise programs" in riseToImperative conversion tests

    //id
    rise.lowering_unit {
        %42 = rise.literal #rise.lit<42.0>
        %id = rise.lambda (%i : !rise.scalar<f32>) -> !rise.scalar<f32> {
            %result = rise.embed(%i) {
                rise.return %i : f32
            } : !rise.scalar<f32>
            rise.return %result : !rise.scalar<f32>
        }
        %result = rise.apply %id, %42
        rise.return
    }

    // ---

    // scalar addition
    rise.lowering_unit {
        %float0 = rise.literal #rise.lit<7.0>
        %float1 = rise.literal #rise.lit<13.0>

        %addFun = rise.lambda (%summand0 : !rise.scalar<f32>, %summand1 : !rise.scalar<f32>) -> !rise.scalar<f32> {
            %result = rise.embed(%summand0, %summand1) {
                %sum = addf %summand0, %summand1 : f32
                rise.return %sum : f32
            } : !rise.scalar<f32>
            rise.return %result : !rise.scalar<f32>
        }
        %result = rise.apply %addFun, %float0, %float1
        rise.return
    }

    // ---

    // tuples
    rise.lowering_unit {
        //creating a simple tuple of an int and a float
        %float0 = rise.literal #rise.lit<7.0>
        %float1 = rise.literal #rise.lit<13.0>

        %tupleFun = rise.tuple #rise.scalar<f32> #rise.scalar<f32>
        %tuple = rise.apply %tupleFun, %float0, %float1
        rise.return
    }

