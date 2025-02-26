package NeuralNetwork.Layers.LossLayer;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import Utilities.CustomMath;

public class Accuracy extends AbstractLossLayer {
    @Override
    public Batch backward(final Batch predictedBatch, final Batch targetBatch) {
        throw new UnsupportedOperationException("Backward method of accuracy loss function is not implemented");
    }

    @Override
    protected double calculate(final DataList predictedList, final DataList targetRow) {
        final int targetListArgMax = CustomMath.argMax(targetRow);

        final double targetListMax = targetRow.getValue(targetListArgMax);
        if (Math.abs(1.0 - targetListMax) > 0.0) {
            throw new IllegalArgumentException("Accuracy requires one-hot target list row");
        }

        return this.calculate(predictedList, targetListArgMax);
    }

    @Override
    protected double calculate(final DataList predictedList, final int targetIndex) {
        final int predictedListArgMax = CustomMath.argMax(predictedList);

        if (predictedListArgMax == targetIndex) {
            return 1.0;
        } else {
            return 0.0;
        }
    }
}
