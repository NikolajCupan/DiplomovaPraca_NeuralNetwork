package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.DataRow;

public class Identity implements IActivationFunction {
    @Override
    public DataRow apply(final DataRow inputRow) {
        return inputRow;
    }
}
