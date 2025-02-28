package NeuralNetwork.BuildingBlocks;

import Utilities.CustomMath;

import java.util.Random;

public class Neuron {
    private static final double BIAS_LEARNING_RATE = 0.001;
    private static final double WEIGHTS_LEARNING_RATE = 0.001;

    private static final double RANDOM_VALUES_SCALE = 0.01;

    private double bias;
    private final DataList weights;

    public Neuron(final int weightsSize, final long seed) {
        final Random random = new Random(seed);

        this.bias = 0.0;
        this.weights = new DataList(weightsSize);

        for (int i = 0; i < weightsSize; ++i) {
            this.weights.setValue(
                    i,
                    Neuron.RANDOM_VALUES_SCALE * random.nextGaussian()
            );
        }
    }

    public Neuron(final double bias, final DataList weights) {
        this.bias = bias;
        this.weights = weights;
    }

    public void updateBias(final double gradientValue) {
        this.bias += -Neuron.BIAS_LEARNING_RATE * gradientValue;
    }

    public void updateWeights(final DataList gradient) {
        for (int i = 0; i < this.weights.getDataListSize(); ++i) {
            final double originalWeightValue = this.weights.getValue(i);
            final double updatedWeightValue = originalWeightValue + (-Neuron.WEIGHTS_LEARNING_RATE * gradient.getValue(i));

            this.weights.setValue(
                    i,
                    updatedWeightValue
            );
        }
    }

    public double forward(final DataList inputRow) {
        if (this.weights.getDataListSize() != inputRow.getDataListSize()) {
           throw new IllegalArgumentException("Size of input row [" + inputRow.getDataListSize() + "] is not equal to size of weights [" + this.weights.getDataListSize() + "]");
        }

        final double dotProduct = CustomMath.dotProduct(this.weights.getDataListRawValues(), inputRow.getDataListRawValues());
        return dotProduct + this.bias;
    }

    public int getWeightsSize() {
        return this.weights.getDataListSize();
    }

    public double getWeight(final int weightIndex) {
        return this.weights.getValue(weightIndex);
    }

    @Override
    public String toString() {
        final StringBuilder builder = new StringBuilder();

        builder.append("Neuron: bias [");
        builder.append(this.bias);
        builder.append("], weights ");
        builder.append(this.weights);

        return builder.toString();
    }
}
