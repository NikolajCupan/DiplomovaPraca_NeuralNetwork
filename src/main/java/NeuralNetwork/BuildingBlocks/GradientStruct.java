package NeuralNetwork.BuildingBlocks;

import java.util.Optional;

public class GradientStruct {
    private Optional<Batch> gradientWithRespectToBiases;
    private Optional<Batch> gradientWithRespectToWeights;
    private Optional<Batch> gradientWithRespectToInputs;

    public GradientStruct() {
        this.gradientWithRespectToBiases = Optional.empty();
        this.gradientWithRespectToWeights = Optional.empty();
        this.gradientWithRespectToInputs = Optional.empty();
    }

    public boolean gradientStructIsEmpty() {
        return this.gradientWithRespectToBiases.isEmpty()
                && this.gradientWithRespectToWeights.isEmpty()
                && this.gradientWithRespectToInputs.isEmpty();
    }

    public Batch getGradientWithRespectToBiases() {
        if (this.gradientWithRespectToBiases.isEmpty()) {
            throw new IllegalArgumentException("Gradient with respect to biases is empty");
        }

        return this.gradientWithRespectToBiases.get();
    }

    public Batch getGradientWithRespectToWeights() {
        if (this.gradientWithRespectToWeights.isEmpty()) {
            throw new IllegalArgumentException("Gradient with respect to weights is empty");
        }

        return this.gradientWithRespectToWeights.get();
    }

    public Batch getGradientWithRespectToInputs() {
        if (this.gradientWithRespectToInputs.isEmpty()) {
            throw new IllegalArgumentException("Gradient with respect to inputs is empty");
        }

        return this.gradientWithRespectToInputs.get();
    }

    public void setGradientWithRespectToBiases(final Batch gradientWithRespectToBiases) {
        if (this.gradientWithRespectToBiases.isPresent()) {
            throw new IllegalArgumentException("Gradient with respect to biases is already set");
        }

        this.gradientWithRespectToBiases = Optional.of(gradientWithRespectToBiases);
    }

    public void setGradientWithRespectToWeights(final Batch gradientWithRespectToWeights) {
        if (this.gradientWithRespectToWeights.isPresent()) {
            throw new IllegalArgumentException("Gradient with respect to weights is already set");
        }

        this.gradientWithRespectToWeights = Optional.of(gradientWithRespectToWeights);
    }

    public void setGradientWithRespectToInputs(final Batch gradientWithRespectToInputs) {
        if (this.gradientWithRespectToInputs.isPresent()) {
            throw new IllegalArgumentException("Gradient with respect to inputs is already set");
        }

        this.gradientWithRespectToInputs = Optional.of(gradientWithRespectToInputs);
    }

    @Override
    public String toString() {
        final StringBuilder builder = new StringBuilder();

        builder.append("{\n");

        builder.append("\tGradient with respect to biases:\n");
        if (this.gradientWithRespectToBiases.isEmpty()) {
            builder.append("\t\t[EMPTY]\n");
        } else {
            builder.append(this.gradientWithRespectToBiases.get()).append("\n");
        }

        builder.append("\tGradient with respect to weights:\n");
        if (this.gradientWithRespectToWeights.isEmpty()) {
            builder.append("\t\t[EMPTY]\n");
        } else {
            builder.append(this.gradientWithRespectToWeights.get()).append("\n");
        }

        builder.append("\tGradient with respect to inputs:\n");
        if (this.gradientWithRespectToInputs.isEmpty()) {
            builder.append("\t\t[EMPTY]\n");
        } else {
            builder.append(this.gradientWithRespectToInputs.get()).append("\n");
        }

        builder.append("}");

        return builder.toString();
    }
}
