package NeuralNetwork.Layers.ActivationLayer;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import NeuralNetwork.Layers.AbstractLayerBase;
import Utilities.GradientStruct;

public class ActivationLayer extends AbstractLayerBase {
    private final IActivationFunction activationFunction;

    public ActivationLayer(final IActivationFunction activationFunction) {
        super();
        this.activationFunction = activationFunction;
    }

    @Override
    protected DataList forward(final DataList inputRow) {
        return this.activationFunction.apply(inputRow);
    }

    @Override
    public GradientStruct backward(final Batch inputGradientBatch) {
        return this.activationFunction.backward(inputGradientBatch, this.getSavedInputBatch(), this.getSavedOutputBatch());
    }
}
