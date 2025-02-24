package NeuralNetwork;

import Utilities.Math;

public class Neuron {
    private final double bias;
    private final Double[] weights;

    public Neuron(
            final double bias,
            final Double[] weights
    ) {
        this.bias = bias;
        this.weights = weights;
    }

    public double calculateOutput(final DataRow inputRow) {
        if (this.weights.length != inputRow.getDataRowSize()) {
           throw new IllegalArgumentException("Size of input row [" + inputRow.getDataRowSize() + "] is not equal to size of weights [" + this.weights.length + "]");
        }

        final double dotProduct = Math.dotProduct(this.weights, inputRow.getDataRowValues());
        return dotProduct + this.bias;
    }

    public long getWeightsSize() {
        return this.weights.length;
    }
}
