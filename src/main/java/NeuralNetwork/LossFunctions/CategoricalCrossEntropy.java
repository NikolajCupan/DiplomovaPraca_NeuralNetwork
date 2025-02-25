package NeuralNetwork.LossFunctions;

import NeuralNetwork.DataList;
import Utilities.CustomMath;

public class CategoricalCrossEntropy implements ILossFunction {
    private static final double CLAMP_VALUE = 0.0000001;

    @Override
    public double calculate(final DataList predictedList, final DataList targetList) {
        final int targetListArgMax = CustomMath.argMax(targetList);

        final double targetListMax = targetList.getValue(targetListArgMax);
        if (Math.abs(1.0 - targetListMax) > 0.0) {
            throw new IllegalArgumentException("Categorical cross entropy requires one-hot target list row");
        }

        return this.calculate(predictedList, targetListArgMax);
    }

    @Override
    public double calculate(final DataList predictedList, final int targetIndex) {
        final double predictedValue = predictedList.getValue(targetIndex);
        final double clampedValue = Math.clamp(predictedValue, CLAMP_VALUE, 1.0 - CLAMP_VALUE);

        return -Math.log(clampedValue);
    }
}
