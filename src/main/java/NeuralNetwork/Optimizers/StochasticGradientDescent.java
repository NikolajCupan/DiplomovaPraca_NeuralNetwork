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
    private final double biasesLearningRate;
    private final double weightsLearningRate;

    public StochasticGradientDescent(final NeuralNetwork neuralNetwork, final double biasesLearningRate, final double weightsLearningRate) {
        this.neuralNetwork = neuralNetwork;
        this.biasesLearningRate = biasesLearningRate;
        this.weightsLearningRate = weightsLearningRate;
    }

    public void optimize() {
        if (!this.neuralNetwork.isBackwardStepExecuted()) {
            throw new RuntimeException("Cannot perform optimization step before backward step");
        }

        final List<LayerBase> layers = this.neuralNetwork.getLayers();

        for (final LayerBase layer : layers) {
            if (layer instanceof final HiddenLayer hiddenLayer) {
                this.optimizeBiases(hiddenLayer);
                this.optimizeWeights(hiddenLayer);
            }
        }
    }

    private void optimizeBiases(final HiddenLayer hiddenLayer) {
        final DataList gradientWRTBiases = hiddenLayer.getSavedOutputGradientStruct().getGradientWithRespectToBiases().getRow(0);
        final List<Neuron> neurons = hiddenLayer.getNeurons();

        for (int neuronIndex = 0; neuronIndex < neurons.size(); ++neuronIndex) {
            final Neuron neuron = neurons.get(neuronIndex);

            final double neuronBiasValue = neuron.getBias();
            final double gradientValue = gradientWRTBiases.getValue(neuronIndex);

            final double optimizedNeuronBiasValue =
                    neuronBiasValue - (this.biasesLearningRate * gradientValue);

            neuron.setBias(optimizedNeuronBiasValue);
        }
    }

    private void optimizeWeights(final HiddenLayer hiddenLayer) {
        final Batch gradientWRTWeights = hiddenLayer.getSavedOutputGradientStruct().getGradientWithRespectToWeights();
        final List<Neuron> neurons = hiddenLayer.getNeurons();

        for (int neuronIndex = 0; neuronIndex < neurons.size(); ++neuronIndex) {
            final Neuron neuron = neurons.get(neuronIndex);

            final DataList neuronWeights = neuron.getWeights();
            final DataList gradient = gradientWRTWeights.getColumn(neuronIndex);

            for (int i = 0; i < neuronWeights.getDataListSize(); ++i) {
                final double originalWeightValue = neuronWeights.getValue(i);
                final double updatedWeightValue = originalWeightValue - this.weightsLearningRate * gradient.getValue(i);

                neuronWeights.setValue(
                        i,
                        updatedWeightValue
                );
            }
        }
    }
}
