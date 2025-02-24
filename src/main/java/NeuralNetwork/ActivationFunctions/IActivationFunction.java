package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.DataRow;

public interface IActivationFunction {
    DataRow apply(final DataRow inputRow);
}
