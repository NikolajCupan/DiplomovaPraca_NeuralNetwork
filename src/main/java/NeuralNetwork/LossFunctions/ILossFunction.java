package NeuralNetwork.LossFunctions;

import NeuralNetwork.DataRow;

public interface ILossFunction {
    double calculate(final DataRow predictedRow, final DataRow targetRow);
}
