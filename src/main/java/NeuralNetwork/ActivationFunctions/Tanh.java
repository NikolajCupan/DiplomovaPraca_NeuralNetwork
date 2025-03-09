package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.BuildingBlocks.DataList;

public class Tanh implements IActivationFunction {
    @Override
    public DataList activate(final DataList inputList) {
        final DataList outputList = new DataList(inputList.getDataListSize());

        for (int i = 0; i < inputList.getDataListSize(); ++i) {
            outputList.setValue(
                    i,
                    Math.tanh(inputList.getValue(i))
            );
        }

        return outputList;
    }

    @Override
    public double derivative(final double input) {
        final DataList inputWrapper = new DataList(1);
        inputWrapper.setValue(0, input);
        final double output = this.activate(inputWrapper).getValue(0);

        return 1.0 - Math.pow(output, 2.0);
    }

    @Override
    public String toString() {
        return "Tanh";
    }
}
