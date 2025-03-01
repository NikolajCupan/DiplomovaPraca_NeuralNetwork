package NeuralNetwork.Optimizers;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.BuildingBlocks.Neuron;
import NeuralNetwork.Layers.Common.HiddenLayer;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.NeuralNetwork;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class StochasticGradientDescentWithMomentum extends OptimizerBase {
    private static class Momentum {
        private DataList biasesMomentum;
        private Batch weightsMomentum;

        private Momentum(final DataList biasesMomentum, final Batch weightsMomentum) {
            this.biasesMomentum = biasesMomentum;
            this.weightsMomentum = weightsMomentum;
        }
    }

    private final double momentum;
    private final Map<Long, Momentum> momentumsMap;

    public StochasticGradientDescentWithMomentum(
            final NeuralNetwork neuralNetwork,
            final double startingLearningRate,
            final double learningRateDecay,
            final double momentum
    ) {
        super(neuralNetwork, startingLearningRate, learningRateDecay);

        this.momentum = momentum;
        this.momentumsMap = new HashMap<>();

        for (final LayerBase layer : neuralNetwork.getLayers()) {
            if (layer instanceof final HiddenLayer hiddenLayer) {
                final int neuronsSize = hiddenLayer.getNeuronsSize();
                final int weightsSize = hiddenLayer.getWeightsSize();

                final DataList biasesMomentum = new DataList(neuronsSize);
                biasesMomentum.fill(0.0);

                // Transposed shape compared to neurons and their weights
                final Batch weightsMomentum = new Batch(weightsSize, neuronsSize);

                this.momentumsMap.put(
                        hiddenLayer.getId(),
                        new Momentum(biasesMomentum, weightsMomentum)
                );
            }
        }
    }

    @Override
    public void optimize() {
        final NeuralNetwork neuralNetwork = this.getNeuralNetwork();
        final List<LayerBase> layers = neuralNetwork.getLayers();

        for (final LayerBase layer : layers) {
            if (layer instanceof final HiddenLayer hiddenLayer) {
                final Momentum layerMomentum = this.momentumsMap.get(hiddenLayer.getId());

                layerMomentum.biasesMomentum = this.getUpdatedBiasesMomentum(hiddenLayer, layerMomentum.biasesMomentum);
                layerMomentum.weightsMomentum = this.getUpdatedWeightsMomentum(hiddenLayer,  layerMomentum.weightsMomentum);

                this.optimizeBiasesAndWeights(hiddenLayer, layerMomentum.biasesMomentum, layerMomentum.weightsMomentum);
            }
        }
    }

    private DataList getUpdatedBiasesMomentum(final HiddenLayer layer, final DataList biasesMomentum) {
        final double currentLearningRate = this.getCurrentLearningRate();

        final DataList gradientWRTBiases = layer.getSavedOutputGradientStruct().getGradientWithRespectToBiases().getRow(0);
        final DataList updatedBiasesMomentum = new DataList(biasesMomentum.getDataListRawValues());

        for (int i = 0; i < biasesMomentum.getDataListSize(); ++i) {
            final double updatedValue =
                    this.momentum * biasesMomentum.getValue(i) - currentLearningRate * gradientWRTBiases.getValue(i);
            updatedBiasesMomentum.setValue(i, updatedValue);
        }

        return updatedBiasesMomentum;
    }

    private Batch getUpdatedWeightsMomentum(final HiddenLayer layer, final Batch weightsMomentum) {
        final double currentLearningRate = this.getCurrentLearningRate();

        final Batch gradientWRTWeights = layer.getSavedOutputGradientStruct().getGradientWithRespectToWeights();
        final Batch updatedWeightsMomentum = new Batch(weightsMomentum.getRowsSize(), weightsMomentum.getColumnsSize());

        for (int rowIndex = 0; rowIndex < weightsMomentum.getRowsSize(); ++rowIndex) {
            final DataList updatedMomentumRow = updatedWeightsMomentum.getRow(rowIndex);
            final DataList momentumRow = weightsMomentum.getRow(rowIndex);
            final DataList gradientRow = gradientWRTWeights.getRow(rowIndex);

            for (int i = 0; i < updatedMomentumRow.getDataListSize(); ++i) {
                final double updatedValue =
                        this.momentum * momentumRow.getValue(i) - currentLearningRate * gradientRow.getValue(i);
                updatedMomentumRow.setValue(i, updatedValue);
            }
        }

        return updatedWeightsMomentum;
    }

    private void optimizeBiasesAndWeights(final HiddenLayer layer, final DataList biasesMomentum, final Batch weightsMomentum) {
        final List<Neuron> neurons = layer.getNeurons();

        for (int neuronIndex = 0; neuronIndex < layer.getNeuronsSize(); ++neuronIndex) {
            final Neuron neuron = neurons.get(neuronIndex);

            final double updatedBias = neuron.getBias() + biasesMomentum.getValue(neuronIndex);
            neuron.setBias(updatedBias);

            final DataList neuronWeights = neuron.getWeights();
            final DataList neuronWeightsMomentum = weightsMomentum.getColumn(neuronIndex);

            for (int weightIndex = 0; weightIndex < neuronWeights.getDataListSize(); ++weightIndex) {
                final double originalWeight = neuronWeights.getValue(weightIndex);
                final double updatedWeight = originalWeight + neuronWeightsMomentum.getValue(weightIndex);

                neuronWeights.setValue(weightIndex, updatedWeight);
            }
        }
    }
}
