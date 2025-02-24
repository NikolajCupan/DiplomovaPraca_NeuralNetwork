package NeuralNetwork.Layers;

import NeuralNetwork.ActivationFunctions.IActivationFunction;
import NeuralNetwork.DataRow;

public class ActivationLayer implements ILayer {
    private final IActivationFunction activationFunction;

    public ActivationLayer(final IActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public DataRow calculateOutputRow(final DataRow inputRow) {
        return this.activationFunction.apply(inputRow);
    }
}
