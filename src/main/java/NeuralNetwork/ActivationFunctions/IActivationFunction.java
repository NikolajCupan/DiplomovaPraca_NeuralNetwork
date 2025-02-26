package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.DataList;

public interface IActivationFunction {
    DataList apply(final DataList inputList);
    double calculateDerivative(final double value);
}
