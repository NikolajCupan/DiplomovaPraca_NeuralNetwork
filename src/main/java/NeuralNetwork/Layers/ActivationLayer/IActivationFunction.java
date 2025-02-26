package NeuralNetwork.Layers.ActivationLayer;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import Utilities.GradientStruct;

public interface IActivationFunction {
    GradientStruct backward(final Batch inputGradientBatch, final Batch savedInputBatch, final Batch savedOutputBatch);

    DataList apply(final DataList inputList);
}
