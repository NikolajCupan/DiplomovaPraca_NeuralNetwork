package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.BuildingBlocks.DataList;

public class Sigmoid implements IActivationFunction {
    @Override
    public DataList activate(final DataList inputList) {
        final DataList outputList = new DataList(inputList.getDataListSize());

        for (int i = 0; i < inputList.getDataListSize(); ++i) {
            outputList.setValue(
                    i,
                    1.0 / (1.0 + Math.exp(-inputList.getValue(i)))
            );
        }

        return outputList;
    }

    @Override
    public double derivative(final double input) {
        final DataList inputWrapper = new DataList(1);
        inputWrapper.setValue(0, input);
        final double output = this.activate(inputWrapper).getValue(0);

        return (1.0 - output) * output;
    }

    @Override
    public String toString() {
        return "Sigmoid";
    }
}
