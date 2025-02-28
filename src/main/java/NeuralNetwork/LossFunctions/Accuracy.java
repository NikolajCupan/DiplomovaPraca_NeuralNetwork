package NeuralNetwork.LossFunctions;

import NeuralNetwork.BuildingBlocks.DataList;
import Utilities.CustomMath;

public class Accuracy implements ILossFunction {
    @Override
    public double loss(final DataList predictedRow, final DataList targetRow) {
        if (predictedRow.getDataListSize() != targetRow.getDataListSize()) {
            throw new IllegalArgumentException("Predicted row size [" + predictedRow.getDataListSize() + "] is not equal to target row size [" + targetRow.getDataListSize() + "]");
        }

        final int targetRowArgMax = CustomMath.argMax(targetRow);

        final double targetRowMax = targetRow.getValue(targetRowArgMax);
        if (Math.abs(1.0 - targetRowMax) > 0.0) {
            throw new IllegalArgumentException("Accuracy requires one-hot target row");
        }

        final int predictedRowArgMax = CustomMath.argMax(predictedRow);

        return predictedRowArgMax == targetRowArgMax ? 1.0 : 0.0;
    }

    @Override
    public String toString() {
        return "Accuracy";
    }
}
