package NeuralNetwork.Optimizers;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.BuildingBlocks.Neuron;
import NeuralNetwork.Layers.Common.HiddenLayer;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.NeuralNetwork;

import java.util.List;

public class StochasticGradientDescent {
    private final NeuralNetwork neuralNetwork;

    private final double biasesStartingLearningRate;
    private final double biasesLearningRateDecay;

    private final double weightsStartingLearningRate;
    private final double weightsLearningRateDecay;

    private int currentIteration;

    public StochasticGradientDescent(
            final NeuralNetwork neuralNetwork,
            final double biasesStartingLearningRate,
            final double biasesLearningRateDecay,
            final double weightsStartingLearningRate,
            final double weightsLearningRateDecay
    ) {
        this.neuralNetwork = neuralNetwork;

        this.biasesStartingLearningRate = biasesStartingLearningRate;
        this.biasesLearningRateDecay = biasesLearningRateDecay;

        this.weightsStartingLearningRate = weightsStartingLearningRate;
        this.weightsLearningRateDecay = weightsLearningRateDecay;

        this.currentIteration = 0;
    }

    public void optimize() {
        if (!this.neuralNetwork.isBackwardStepExecuted()) {
            throw new RuntimeException("Cannot perform optimization step before backward step");
        }

        final List<LayerBase> layers = this.neuralNetwork.getLayers();

        final double currentBiasesLearningRate = this.getCurrentBiasesLearningRate();
        final double currentWeightsLearningRate = this.getCurrentWeightsLearningRate();

        for (final LayerBase layer : layers) {
            if (layer instanceof final HiddenLayer hiddenLayer) {
                StochasticGradientDescent.optimizeBiases(hiddenLayer, currentBiasesLearningRate);
                StochasticGradientDescent.optimizeWeights(hiddenLayer, currentWeightsLearningRate);
            }
        }

        ++this.currentIteration;
    }

    public double getCurrentBiasesLearningRate() {
        return StochasticGradientDescent.getCurrentLearningRate(this.biasesStartingLearningRate, this.biasesLearningRateDecay, this.currentIteration);
    }

    public double getCurrentWeightsLearningRate() {
        return StochasticGradientDescent.getCurrentLearningRate(this.weightsStartingLearningRate, this.weightsLearningRateDecay, this.currentIteration);
    }

    private static double getCurrentLearningRate(
            final double startingLearningRate,
            final double learningRateDecay,
            final int currentIteration
    ) {
        return startingLearningRate * (1.0 / (1.0 + learningRateDecay * currentIteration));
    }

    private static void optimizeBiases(final HiddenLayer hiddenLayer, final double biasesLearningRate) {
        final DataList gradientWRTBiases = hiddenLayer.getSavedOutputGradientStruct().getGradientWithRespectToBiases().getRow(0);
        final List<Neuron> neurons = hiddenLayer.getNeurons();

        for (int neuronIndex = 0; neuronIndex < neurons.size(); ++neuronIndex) {
            final Neuron neuron = neurons.get(neuronIndex);

            final double neuronBiasValue = neuron.getBias();
            final double gradientValue = gradientWRTBiases.getValue(neuronIndex);

            final double optimizedNeuronBiasValue =
                    neuronBiasValue - (biasesLearningRate * gradientValue);

            neuron.setBias(optimizedNeuronBiasValue);
        }
    }

    private static void optimizeWeights(final HiddenLayer hiddenLayer, final double weightsLearningRate) {
        final Batch gradientWRTWeights = hiddenLayer.getSavedOutputGradientStruct().getGradientWithRespectToWeights();
        final List<Neuron> neurons = hiddenLayer.getNeurons();

        for (int neuronIndex = 0; neuronIndex < neurons.size(); ++neuronIndex) {
            final Neuron neuron = neurons.get(neuronIndex);

            final DataList neuronWeights = neuron.getWeights();
            final DataList gradient = gradientWRTWeights.getColumn(neuronIndex);

            for (int i = 0; i < neuronWeights.getDataListSize(); ++i) {
                final double originalWeightValue = neuronWeights.getValue(i);
                final double updatedWeightValue = originalWeightValue - weightsLearningRate * gradient.getValue(i);

                neuronWeights.setValue(
                        i,
                        updatedWeightValue
                );
            }
        }
    }
}
