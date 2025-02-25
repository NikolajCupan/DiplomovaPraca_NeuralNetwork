package NeuralNetwork.LossFunctions;

import NeuralNetwork.DataRow;
import Utilities.CustomMath;

public class CategoricalCrossEntropy implements ILossFunction {
    private static final double CLAMP_VALUE = 0.0000001;

    @Override
    public double calculate(final DataRow predictedRow, final DataRow targetRow) {
        final int targetRowArgMax = CustomMath.argMax(targetRow);

        final double targetRowMax = targetRow.getValue(targetRowArgMax);
        if (Math.abs(1.0 - targetRowMax) > 0.0) {
            throw new IllegalArgumentException("Categorical cross entropy requires one-hot target data row");
        }

        return this.calculate(predictedRow, targetRowArgMax);
    }

    @Override
    public double calculate(final DataRow predictedRow, final int targetIndex) {
        final double predictedValue = predictedRow.getValue(targetIndex);
        final double clampedValue = Math.clamp(predictedValue, CLAMP_VALUE, 1.0 - CLAMP_VALUE);

        return -Math.log(clampedValue);
    }
}
