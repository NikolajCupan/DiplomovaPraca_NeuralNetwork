package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.BuildingBlocks.DataList;

public class LeakyRectifiedLinearUnit implements IActivationFunction {
    private final double slope;

    public LeakyRectifiedLinearUnit(final double slope) {
        this.slope = slope;
    }

    @Override
    public DataList activate(final DataList inputList) {
        final DataList outputList = new DataList(inputList.getDataListSize());

        for (int i = 0; i < inputList.getDataListSize(); ++i) {
            final double value = inputList.getValue(i);
            double calculatedValue;

            if (value >= 0) {
                calculatedValue = value;
            } else {
                calculatedValue = this.slope * value;
            }

            outputList.setValue(i, calculatedValue);
        }

        return outputList;
    }

    @Override
    public double derivative(final double input) {
        if (input >= 0.0) {
            return 1.0;
        } else {
            return this.slope;
        }
    }

    @Override
    public String toString() {
        return "Leaky rectified linear unit";
    }
}
