package NeuralNetwork.LossFunctions;

import NeuralNetwork.DataRow;
import Utilities.CustomMath;

public class Accuracy implements ILossFunction {
    @Override
    public double calculate(final DataRow predictedRow, final DataRow targetRow) {
        final int targetRowArgMax = CustomMath.argMax(targetRow);

        final double targetRowMax = targetRow.getValue(targetRowArgMax);
        if (Math.abs(1.0 - targetRowMax) > 0.0) {
            throw new IllegalArgumentException("Accuracy requires one-hot target data row");
        }

        return this.calculate(predictedRow, targetRowArgMax);
    }

    @Override
    public double calculate(final DataRow predictedRow, final int targetIndex) {
        final int predictedRowArgMax = CustomMath.argMax(predictedRow);

        if (predictedRowArgMax == targetIndex) {
            return 1.0;
        } else {
            return 0.0;
        }
    }
}
