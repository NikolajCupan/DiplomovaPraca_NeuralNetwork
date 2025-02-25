package NeuralNetwork.LossFunctions;

import NeuralNetwork.DataRow;
import Utilities.CustomMath;

public class CategoricalCrossEntropy implements ILossFunction {
    @Override
    public Double calculate(final DataRow predictedRow, final DataRow targetRow) {
        final int argMax = CustomMath.argMax(targetRow);
        final Double max = targetRow.getValue(argMax);

        if (Math.abs(1.0 - max) > 0.0) {
            throw new IllegalArgumentException("Categorical cross entropy requires one-hot target data row");
        }

        return -Math.log(predictedRow.getValue(argMax));
    }
}
