public class Neuron {
    private final double bias;
    private final Double[] weights;

    private final long weightsSize;

    public Neuron(
            final double bias,
            final Double[] weights
    ) {
        this.bias = bias;
        this.weights = weights;

        this.weightsSize = weights.length;
    }

    public double calculateOutput(final Double[] inputs) {
        if (this.weightsSize != inputs.length) {
           throw new IllegalArgumentException("Size of inputs [" + inputs.length + "] is not equal to size of weights [" + this.weightsSize + "]");
        }

        final double dotProduct = Math.dotProduct(this.weights, inputs);
        return dotProduct + this.bias;
    }

    public long getWeightsSize() {
        return this.weightsSize;
    }
}
