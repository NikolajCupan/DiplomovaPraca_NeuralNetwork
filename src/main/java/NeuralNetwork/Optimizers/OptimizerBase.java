package NeuralNetwork.Optimizers;

import NeuralNetwork.NeuralNetwork;

public abstract class OptimizerBase {
    private final NeuralNetwork neuralNetwork;

    private final double startingLearningRate;
    private final double learningRateDecay;

    private int currentIteration;

    protected OptimizerBase(
            final NeuralNetwork neuralNetwork,
            final double startingLearningRate,
            final double learningRateDecay
    ) {
        this.neuralNetwork = neuralNetwork;

        this.startingLearningRate = startingLearningRate;
        this.learningRateDecay = learningRateDecay;

        this.currentIteration = 0;
    }

    public void performOptimization() {
        this.optimize();
        ++this.currentIteration;
    }

    protected NeuralNetwork getNeuralNetwork() {
        return this.neuralNetwork;
    }

    public double getCurrentLearningRate() {
        return this.startingLearningRate * (1.0 / (1.0 + this.learningRateDecay * this.currentIteration));
    }

    public int getCurrentIteration() {
        return this.currentIteration;
    }

    protected abstract void optimize();
}
