public class Neuron {
    private final double bias;
    private final Double[] weights;

    private final int weightsLength;

    public Neuron(
            final double bias,
            final Double[] weights
    ) {
        this.bias = bias;
        this.weights = weights;

        this.weightsLength = weights.length;
    }

    public double calculateOutput(final Double[] inputs) {
        if (this.weightsLength != inputs.length) {
           throw new IllegalArgumentException("Length of inputs [" + inputs.length + "] is not equal to length of weights [" + this.weightsLength + "]");
        }

        final double dotProduct = Math.dotProduct(this.weights, inputs);
        return dotProduct + this.bias;
    }
}
