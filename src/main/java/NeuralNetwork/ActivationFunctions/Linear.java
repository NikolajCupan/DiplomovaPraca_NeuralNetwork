package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.BuildingBlocks.DataList;

public class Linear implements IActivationFunction {
    @Override
    public DataList activate(final DataList inputList) {
        final DataList outputList = new DataList(inputList.getDataListSize());

        for (int i = 0; i < inputList.getDataListSize(); ++i) {
            outputList.setValue(
                    i,
                    inputList.getValue(i)
            );
        }

        return outputList;
    }

    @Override
    public double derivative(final double input) {
        return input;
    }

    @Override
    public String toString() {
        return "Linear";
    }
}
