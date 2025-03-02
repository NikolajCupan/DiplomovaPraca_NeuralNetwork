package NeuralNetwork.BuildingBlocks;

import Utilities.CustomMath;

import java.util.Random;

public class Neuron {
    private static final double RANDOM_VALUES_SCALE = 0.01;

    private double bias;
    private final DataList weights;

    public Neuron(final int weightsSize) {
        this.bias = 0.0;
        this.weights = new DataList(weightsSize);

        final Random random = new Random();
        this.initializeWeightsRandomly(weightsSize, random);
    }

    public Neuron(final int weightsSize, final long seed) {
        this.bias = 0.0;
        this.weights = new DataList(weightsSize);

        final Random random = new Random(seed);
        this.initializeWeightsRandomly(weightsSize, random);
    }

    private void initializeWeightsRandomly(final int weightsSize, final Random random) {
        for (int i = 0; i < weightsSize; ++i) {
            this.weights.setValue(
                    i,
                    // 25
                    Neuron.RANDOM_VALUES_SCALE * random.nextGaussian()
            );
        }
    }

    public Neuron(final double bias, final DataList weights) {
        this.bias = bias;
        this.weights = weights;
    }

    public double getBias() {
        return this.bias;
    }

    public int getWeightsSize() {
        return this.weights.getDataListSize();
    }

    public double getWeight(final int weightIndex) {
        return this.weights.getValue(weightIndex);
    }

    public DataList getWeights() {
        return this.weights;
    }

    public void setBias(final double bias) {
        this.bias = bias;
    }

    public double forward(final DataList inputRow) {
        if (this.weights.getDataListSize() != inputRow.getDataListSize()) {
            throw new IllegalArgumentException("Size of input row [" + inputRow.getDataListSize() + "] is not equal to size of weights [" + this.weights.getDataListSize() + "]");
        }

        final double dotProduct = CustomMath.dotProduct(this.weights.getDataListRawValues(), inputRow.getDataListRawValues());
        return dotProduct + this.bias;
    }

    @Override
    public String toString() {
        return "Neuron: bias [" +
                this.bias +
                "], weights " +
                this.weights;
    }
}
