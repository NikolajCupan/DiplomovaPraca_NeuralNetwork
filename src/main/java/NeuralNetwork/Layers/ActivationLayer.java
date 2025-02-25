package NeuralNetwork.Layers;

import NeuralNetwork.ActivationFunctions.IActivationFunction;
import NeuralNetwork.DataList;

public class ActivationLayer extends AbstractLayer {
    private final IActivationFunction activationFunction;

    public ActivationLayer(final IActivationFunction activationFunction) {
        super();
        this.activationFunction = activationFunction;
    }

    @Override
    protected DataList calculateOutputRow(final DataList inputRow) {
        return this.activationFunction.apply(inputRow);
    }
}
