package NeuralNetwork;

import Utilities.CustomMath;

import java.util.Random;

public class Neuron {
    private static final double RANDOM_VALUES_SCALE = 0.01;

    private final double bias;
    private final DataRow weights;

    public Neuron(final int weightsSize, final long seed) {
        final Random random = new Random(seed);

        this.bias = 0.0;
        this.weights = new DataRow(weightsSize);

        for (int i = 0; i < weightsSize; ++i) {
            this.weights.setValue(
                    i,
                    Neuron.RANDOM_VALUES_SCALE * random.nextGaussian()
            );
        }
    }

    public Neuron(final double bias, final DataRow weights) {
        this.bias = bias;
        this.weights = weights;
    }

    public double calculateOutput(final DataRow inputRow) {
        if (this.weights.getDataRowSize() != inputRow.getDataRowSize()) {
           throw new IllegalArgumentException("Size of input row [" + inputRow.getDataRowSize() + "] is not equal to size of weights [" + this.weights.getDataRowSize() + "]");
        }

        final double dotProduct = CustomMath.dotProduct(this.weights.getDataRowValues(), inputRow.getDataRowValues());
        return dotProduct + this.bias;
    }

    public int getWeightsSize() {
        return this.weights.getDataRowSize();
    }

    public double getWeight(final int weightIndex) {
        return this.weights.getValue(weightIndex);
    }
}
