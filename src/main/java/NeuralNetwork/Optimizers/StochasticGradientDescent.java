package NeuralNetwork.Optimizers;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.BuildingBlocks.Neuron;
import NeuralNetwork.Layers.Common.HiddenLayer;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.NeuralNetwork;

import java.util.List;

public class StochasticGradientDescent extends OptimizerBase {
    public StochasticGradientDescent(
            final NeuralNetwork neuralNetwork,
            final double startingLearningRate,
            final double learningRateDecay
    ) {
        super(neuralNetwork, startingLearningRate, learningRateDecay);
    }

    @Override
    protected void optimize() {
        final NeuralNetwork neuralNetwork = this.getNeuralNetwork();

        final List<LayerBase> layers = neuralNetwork.getLayers();
        final double currentLearningRate = this.getCurrentLearningRate();

        for (final LayerBase layer : layers) {
            if (layer instanceof final HiddenLayer hiddenLayer) {
                this.optimizeBiases(hiddenLayer, currentLearningRate);
                this.optimizeWeights(hiddenLayer, currentLearningRate);
            }
        }
    }

    private void optimizeBiases(final HiddenLayer hiddenLayer, final double learningRate) {
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

    private void optimizeWeights(final HiddenLayer hiddenLayer, final double learningRate) {
        final Batch gradientWRTWeights = hiddenLayer.getSavedOutputGradientStruct().getGradientWithRespectToWeights();
        final List<Neuron> neurons = hiddenLayer.getNeurons();

        for (int neuronIndex = 0; neuronIndex < neurons.size(); ++neuronIndex) {
            final Neuron neuron = neurons.get(neuronIndex);

            final DataList neuronWeights = neuron.getWeights();
            final DataList gradient = gradientWRTWeights.getColumn(neuronIndex);

            for (int i = 0; i < neuronWeights.getDataListSize(); ++i) {
                final double originalWeightValue = neuronWeights.getValue(i);
                final double updatedWeightValue = originalWeightValue - (learningRate * gradient.getValue(i));

                neuronWeights.setValue(
                        i,
                        updatedWeightValue
                );
            }
        }
    }
}
