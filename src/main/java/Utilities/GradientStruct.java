package Utilities;

import NeuralNetwork.Batch;

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
}
