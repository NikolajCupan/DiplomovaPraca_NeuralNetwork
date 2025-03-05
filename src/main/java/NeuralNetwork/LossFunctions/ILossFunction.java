package NeuralNetwork.LossFunctions;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import Utilities.CustomMath;

public interface ILossFunction {
    default double getLossForPrinting(final Batch savedOutputBatch) {
        final DataList savedOutput = savedOutputBatch.getRow(0);
        return CustomMath.mean(savedOutput);
    }

    // "inputBatch" can be for example output from Linear activation layer,
    // that means in loss layer it is an input
    Batch backward(final Batch inputBatch, final Batch targetBatch);

    double getAccuracyForPrinting(final Batch predictedBatch, final Batch targetBatch);
    double loss(final DataList predictedRow, final DataList targetRow);
}
