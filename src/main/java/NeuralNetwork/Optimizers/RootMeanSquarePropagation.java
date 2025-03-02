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

public class RootMeanSquarePropagation extends OptimizerBase {
    private static class Cache {
        private DataList biasesCache;
        private Batch weightsCache;

        private Cache(final DataList biasesCache, final Batch weightsCache) {
            this.biasesCache = biasesCache;
            this.weightsCache = weightsCache;
        }
    }

    private final double epsilon;
    private final double rho;
    private final Map<Long, Cache> cachesMap;

    public RootMeanSquarePropagation(
            final NeuralNetwork neuralNetwork,
            final double startingLearningRate,
            final double learningRateDecay,
            final double epsilon,
            final double rho
    ) {
        super(neuralNetwork, startingLearningRate, learningRateDecay);

        this.epsilon = epsilon;
        this.rho = rho;
        this.cachesMap = new HashMap<>();

        for (final LayerBase layer : neuralNetwork.getLayers()) {
            if (layer instanceof final HiddenLayer hiddenLayer) {
                final int neuronsSize = hiddenLayer.getNeuronsSize();
                final int weightsSize = hiddenLayer.getWeightsSize();

                final DataList biasesCache = new DataList(neuronsSize);
                biasesCache.fill(0.0);

                // Transposed shape compared to neurons and their weights
                final Batch weightsCache = new Batch(weightsSize, neuronsSize);

                this.cachesMap.put(
                        hiddenLayer.getId(),
                        new Cache(biasesCache, weightsCache)
                );
            }
        }
    }

    @Override
    protected void optimize() {
        final NeuralNetwork neuralNetwork = this.getNeuralNetwork();

        final List<LayerBase> layers = neuralNetwork.getLayers();
        final double currentLearningRate = this.getCurrentLearningRate();

        for (final LayerBase layer : layers) {
            if (layer instanceof final HiddenLayer hiddenLayer) {
                final Cache layerCache = this.cachesMap.get(hiddenLayer.getId());

                layerCache.biasesCache = this.getUpdatedBiasesCache(hiddenLayer, layerCache.biasesCache);
                layerCache.weightsCache = this.getUpdatedWeightsCache(hiddenLayer, layerCache.weightsCache);

                this.optimizeBiasesAndWeights(hiddenLayer, layerCache.biasesCache, layerCache.weightsCache);
            }
        }
    }

    private DataList getUpdatedBiasesCache(final HiddenLayer layer, final DataList biasesCache) {
        final DataList gradientWRTBiases = layer.getSavedOutputGradientStruct().getGradientWithRespectToBiases().getRow(0);
        final DataList updatedBiasesCache = new DataList(gradientWRTBiases.getDataListSize());

        for (int i = 0; i < gradientWRTBiases.getDataListSize(); ++i) {
            final double originalCacheValue = biasesCache.getValue(i);
            final double updatedValue =
                    this.rho * originalCacheValue + (1 - this.rho) * Math.pow(gradientWRTBiases.getValue(i), 2.0);

            updatedBiasesCache.setValue(i, updatedValue);
        }

        return updatedBiasesCache;
    }

    private Batch getUpdatedWeightsCache(final HiddenLayer layer, final Batch weightsCache) {
        final Batch gradientWRTWeights = layer.getSavedOutputGradientStruct().getGradientWithRespectToWeights();
        final Batch updatedWeightsCache = new Batch();

        for (int rowIndex = 0; rowIndex < gradientWRTWeights.getRowsSize(); ++rowIndex) {
            final DataList weightsCacheRow = weightsCache.getRow(rowIndex);
            final DataList gradientRow = gradientWRTWeights.getRow(rowIndex);
            final DataList updatedWeightsRow = new DataList(gradientRow.getDataListSize());

            for (int i = 0; i < gradientRow.getDataListSize(); ++i) {
                final double originalCacheValue = weightsCacheRow.getValue(i);
                final double updatedValue =
                        this.rho * originalCacheValue + (1 - this.rho) * Math.pow(gradientRow.getValue(i), 2.0);

                updatedWeightsRow.setValue(i, updatedValue);
            }

            updatedWeightsCache.addRow(updatedWeightsRow);
        }

        return updatedWeightsCache;
    }

    private void optimizeBiasesAndWeights(final HiddenLayer layer, final DataList biasesCache, final Batch weigtsCache) {
        final double currentLearningRate = this.getCurrentLearningRate();

        final List<Neuron> neurons = layer.getNeurons();
        final DataList gradientWRTBiases = layer.getSavedOutputGradientStruct().getGradientWithRespectToBiases().getRow(0);
        final Batch gradientWRTWeights = layer.getSavedOutputGradientStruct().getGradientWithRespectToWeights();

        for (int neuronIndex = 0; neuronIndex < layer.getNeuronsSize(); ++neuronIndex) {
            final Neuron neuron = neurons.get(neuronIndex);

            final double cachedBias = biasesCache.getValue(neuronIndex);
            final double originalBias = neuron.getBias();
            final double updatedBias =
                    originalBias - (currentLearningRate * gradientWRTBiases.getValue(neuronIndex)) / (Math.sqrt(cachedBias) + this.epsilon);
            neuron.setBias(updatedBias);

            final DataList neuronWeights = neuron.getWeights();
            final DataList neuronWeightsCache = weigtsCache.getColumn(neuronIndex);
            final DataList gradientWeightsColumn = gradientWRTWeights.getColumn(neuronIndex);

            for (int weightIndex = 0; weightIndex < neuronWeights.getDataListSize(); ++weightIndex) {
                final double cachedWeight = neuronWeightsCache.getValue(weightIndex);
                final double originalWeight = neuronWeights.getValue(weightIndex);
                final double updatedWeight =
                        originalWeight - (currentLearningRate * gradientWeightsColumn.getValue(weightIndex)) / (Math.sqrt(cachedWeight) + this.epsilon);

                neuronWeights.setValue(weightIndex, updatedWeight);
            }
        }
    }
}
