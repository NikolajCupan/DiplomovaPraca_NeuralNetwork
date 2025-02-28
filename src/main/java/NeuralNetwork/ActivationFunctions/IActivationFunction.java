package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.BuildingBlocks.DataList;

public interface IActivationFunction {
    default boolean requiresCustomBackwardStep() {
        return false;
    }

    DataList activate(final DataList inputList);
    double derivative(final double input);
}
