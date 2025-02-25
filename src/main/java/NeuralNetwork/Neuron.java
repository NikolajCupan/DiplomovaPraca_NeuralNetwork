package NeuralNetwork;

import Utilities.CustomMath;

import java.util.Random;

public class Neuron {
    private static final double RANDOM_VALUES_SCALE = 0.01;

    private final double bias;
    private final Double[] weights;

    public Neuron(final int weightsSize, final long seed) {
        final Random random = new Random(seed);

        this.bias = 0.0;
        this.weights = new Double[weightsSize];

        for (int i = 0; i < weightsSize; ++i) {
            this.weights[i] = Neuron.RANDOM_VALUES_SCALE * random.nextGaussian();
        }
    }

    public Neuron(final double bias, final Double[] weights) {
        this.bias = bias;
        this.weights = weights;
    }

    public Double calculateOutput(final DataRow inputRow) {
        if (this.weights.length != inputRow.getDataRowSize()) {
           throw new IllegalArgumentException("Size of input row [" + inputRow.getDataRowSize() + "] is not equal to size of weights [" + this.weights.length + "]");
        }

        final Double dotProduct = CustomMath.dotProduct(this.weights, inputRow.getDataRowValues());
        return dotProduct + this.bias;
    }

    public int getWeightsSize() {
        return this.weights.length;
    }
}
