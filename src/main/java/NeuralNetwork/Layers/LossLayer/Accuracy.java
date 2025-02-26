package NeuralNetwork.Layers.LossLayer;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import Utilities.CustomMath;

public class Accuracy extends AbstractLossLayer {
    public Accuracy() {
        super();
    }

    @Override
    public Batch backward() {
        throw new UnsupportedOperationException("Backward method of accuracy loss function is not implemented");
    }

    @Override
    protected double forward(final DataList predictedList, final DataList targetRow) {
        final int targetListArgMax = CustomMath.argMax(targetRow);

        final double targetListMax = targetRow.getValue(targetListArgMax);
        if (Math.abs(1.0 - targetListMax) > 0.0) {
            throw new IllegalArgumentException("Accuracy requires one-hot target list row");
        }

        return this.forward(predictedList, targetListArgMax);
    }

    @Override
    protected double forward(final DataList predictedList, final int targetIndex) {
        final int predictedListArgMax = CustomMath.argMax(predictedList);

        if (predictedListArgMax == targetIndex) {
            return 1.0;
        } else {
            return 0.0;
        }
    }
}
