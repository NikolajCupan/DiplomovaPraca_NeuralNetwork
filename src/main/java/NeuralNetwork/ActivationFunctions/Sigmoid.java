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
    public double derivative(final double value) {
        return 0.0;
    }

    @Override
    public String toString() {
        return "Sigmoid";
    }
}
