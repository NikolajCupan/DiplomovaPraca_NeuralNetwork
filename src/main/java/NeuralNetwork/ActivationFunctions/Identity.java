package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.DataList;

public class Identity implements IActivationFunction {
    @Override
    public DataList apply(final DataList inputList) {
        return inputList;
    }
}
