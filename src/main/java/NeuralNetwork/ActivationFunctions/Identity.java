package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.DataList;

public class Identity implements IActivationFunction {
    @Override
    public DataList apply(final DataList inputList) {
        return inputList;
    }

    @Override
    public double calculateDerivative(final double value) {
        throw new UnsupportedOperationException("Identity activation function derivative is not implemented");
    }
}
