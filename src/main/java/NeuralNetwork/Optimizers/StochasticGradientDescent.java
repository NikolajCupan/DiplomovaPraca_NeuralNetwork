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

    private final double startingLearningRate;
    private final double learningRateDecay;

    private int currentIteration;

    public StochasticGradientDescent(
            final NeuralNetwork neuralNetwork,
            final double startingLearningRate,
            final double learningRateDecay
    ) {
        this.neuralNetwork = neuralNetwork;

        this.startingLearningRate = startingLearningRate;
        this.learningRateDecay = learningRateDecay;

        this.currentIteration = 0;
    }

    public void optimize() {
        if (!this.neuralNetwork.isBackwardStepExecuted()) {
            throw new RuntimeException("Cannot perform optimization step before backward step");
        }

        final List<LayerBase> layers = this.neuralNetwork.getLayers();
        final double currentLearningRate = this.getCurrentLearningRate();

        for (final LayerBase layer : layers) {
            if (layer instanceof final HiddenLayer hiddenLayer) {
                StochasticGradientDescent.optimizeBiases(hiddenLayer, currentLearningRate);
                StochasticGradientDescent.optimizeWeights(hiddenLayer, currentLearningRate);
            }
        }

        ++this.currentIteration;
    }

    public double getCurrentLearningRate() {
        return this.startingLearningRate * (1.0 / (1.0 + this.learningRateDecay * this.currentIteration));
    }

    private static void optimizeBiases(final HiddenLayer hiddenLayer, final double learningRate) {
        final DataList gradientWRTBiases = hiddenLayer.getSavedOutputGradientStruct().getGradientWithRespectToBiases().getRow(0);
        final List<Neuron> neurons = hiddenLayer.getNeurons();

        for (int neuronIndex = 0; neuronIndex < neurons.size(); ++neuronIndex) {
            final Neuron neuron = neurons.get(neuronIndex);

            final double neuronBiasValue = neuron.getBias();
            final double gradientValue = gradientWRTBiases.getValue(neuronIndex);

            final double optimizedNeuronBiasValue =
                    neuronBiasValue - (learningRate * gradientValue);

            neuron.setBias(optimizedNeuronBiasValue);
        }
    }

    private static void optimizeWeights(final HiddenLayer hiddenLayer, final double learningRate) {
        final Batch gradientWRTWeights = hiddenLayer.getSavedOutputGradientStruct().getGradientWithRespectToWeights();
        final List<Neuron> neurons = hiddenLayer.getNeurons();

        for (int neuronIndex = 0; neuronIndex < neurons.size(); ++neuronIndex) {
            final Neuron neuron = neurons.get(neuronIndex);

            final DataList neuronWeights = neuron.getWeights();
            final DataList gradient = gradientWRTWeights.getColumn(neuronIndex);

            for (int i = 0; i < neuronWeights.getDataListSize(); ++i) {
                final double originalWeightValue = neuronWeights.getValue(i);
                final double updatedWeightValue = originalWeightValue - learningRate * gradient.getValue(i);

                neuronWeights.setValue(
                        i,
                        updatedWeightValue
                );
            }
        }
    }
}
