package NeuralNetwork.Layers;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.GradientStruct;

public class InputLayer extends LayerBase {
    public InputLayer() {
        super();
    }

    @Override
    public void forward(final Batch inputBatch) {
        this.setSavedInputBatch(inputBatch);
        this.setSavedOutputBatch(inputBatch);
    }

    @Override
    public void backward(final GradientStruct inputGradientStruct) {
        this.setSavedInputGradientStruct(inputGradientStruct);
        this.setSavedOutputGradientStruct(inputGradientStruct);
    }

    @Override
    public String toString() {
        return "{\n\tInput layer: input\n}";
    }
}
