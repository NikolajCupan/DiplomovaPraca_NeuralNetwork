package NeuralNetwork.LossFunctions;

import NeuralNetwork.BuildingBlocks.DataList;

public interface ILossFunction {
    double loss(final DataList predictedRow, final DataList targetRow);
}
