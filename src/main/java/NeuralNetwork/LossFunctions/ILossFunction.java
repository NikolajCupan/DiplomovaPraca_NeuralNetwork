package NeuralNetwork.LossFunctions;

import NeuralNetwork.DataRow;

public interface ILossFunction {
    Double calculate(final DataRow predictedRow, final DataRow targetRow);
}
