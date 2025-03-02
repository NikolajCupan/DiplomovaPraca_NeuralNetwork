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

public class AdaptiveMomentum extends OptimizerBase {
    private static class Momentum {
        private DataList biasesMomentum;
        private Batch weightsMomentum;

        private Momentum(final DataList biasesMomentum, final Batch weightsMomentum) {
            this.biasesMomentum = biasesMomentum;
            this.weightsMomentum = weightsMomentum;
        }
    }

    private static class Cache {
        private DataList biasesCache;
        private Batch weightsCache;

        private Cache(final DataList biasesCache, final Batch weightsCache) {
            this.biasesCache = biasesCache;
            this.weightsCache = weightsCache;
        }
    }

    private final double epsilon;
    private final double beta1;
    private final double beta2;

    private final Map<Long, Momentum> momentumsMap;
    private final Map<Long, Cache> cachesMap;

    public AdaptiveMomentum(
            final NeuralNetwork neuralNetwork,
            final double startingLearningRate,
            final double learningRateDecay,
            final double epsilon,
            final double beta1,
            final double beta2
    ) {
        super(neuralNetwork, startingLearningRate, learningRateDecay);

        this.epsilon = epsilon;
        this.beta1 = beta1;
        this.beta2 = beta2;

        this.momentumsMap = new HashMap<>();
        this.cachesMap = new HashMap<>();

        for (final LayerBase layer : neuralNetwork.getLayers()) {
            if (layer instanceof final HiddenLayer hiddenLayer) {
                final int neuronsSize = hiddenLayer.getNeuronsSize();
                final int weightsSize = hiddenLayer.getWeightsSize();


                // Momentums
                final DataList biasesMomentum = new DataList(neuronsSize);
                biasesMomentum.fill(0.0);

                // Transposed shape compared to neurons and their weights
                final Batch weightsMomentum = new Batch(weightsSize, neuronsSize);

                this.momentumsMap.put(
                        hiddenLayer.getId(),
                        new Momentum(biasesMomentum, weightsMomentum)
                );


                // Caches
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

        for (final LayerBase layer : layers) {
            if (layer instanceof final HiddenLayer hiddenLayer) {
                final Momentum layerMomentum = this.momentumsMap.get(hiddenLayer.getId());

                layerMomentum.biasesMomentum = this.getUpdatedBiasesMomentum(hiddenLayer, layerMomentum.biasesMomentum);
                layerMomentum.weightsMomentum = this.getUpdatedWeightsMomentum(hiddenLayer, layerMomentum.weightsMomentum);

                final DataList correctedBiasesMomentum = this.getCorrectedBiasesMomentum(layerMomentum.biasesMomentum);
                final Batch correctedWeightsMomentum = this.getCorrectedWeightsMomentum(layerMomentum.weightsMomentum);


                final Cache layerCache = this.cachesMap.get(hiddenLayer.getId());

                layerCache.biasesCache = this.getUpdatedBiasesCache(hiddenLayer, layerCache.biasesCache);
                layerCache.weightsCache = this.getUpdatedWeightsCache(hiddenLayer, layerCache.weightsCache);

                final DataList correctedBiasesCache = this.getCorrectedBiasesCache(layerCache.biasesCache);
                final Batch correctedWeightsCache = this.getCorrectedWeightsCache(layerCache.weightsCache);

                this.optimizeBiasesAndWeights(
                        hiddenLayer, correctedBiasesMomentum, correctedWeightsMomentum, correctedBiasesCache, correctedWeightsCache
                );
            }
        }
    }

    private DataList getUpdatedBiasesMomentum(final HiddenLayer layer, final DataList biasesMomentum) {
        final DataList gradientWRTBiases = layer.getSavedOutputGradientStruct().getGradientWithRespectToBiases().getRow(0);
        final DataList updatedBiasesMomentum = new DataList(biasesMomentum.getDataListSize());

        for (int i = 0; i < biasesMomentum.getDataListSize(); ++i) {
            final double updatedValue =
                    this.beta1 * biasesMomentum.getValue(i) + (1.0 - this.beta1) * gradientWRTBiases.getValue(i);
            updatedBiasesMomentum.setValue(i, updatedValue);
        }

        return updatedBiasesMomentum;
    }

    private DataList getCorrectedBiasesMomentum(final DataList biasesMomentum) {
        final int currentIteration = this.getCurrentIteration();
        final DataList correctedBiasesMomentum = new DataList(biasesMomentum.getDataListSize());

        for (int i = 0; i < biasesMomentum.getDataListSize(); ++i) {
            final double correctedValue =
                    biasesMomentum.getValue(i) / (1.0 - Math.pow(this.beta1, currentIteration + 1.0));
            correctedBiasesMomentum.setValue(i, correctedValue);
        }

        return correctedBiasesMomentum;
    }

    private Batch getUpdatedWeightsMomentum(final HiddenLayer layer, final Batch weightsMomentum) {
        final Batch gradientWRTWeights = layer.getSavedOutputGradientStruct().getGradientWithRespectToWeights();
        final Batch updatedWeightsMomentum = new Batch(weightsMomentum.getRowsSize(), weightsMomentum.getColumnsSize());

        for (int rowIndex = 0; rowIndex < weightsMomentum.getRowsSize(); ++rowIndex) {
            final DataList updatedMomentumRow = updatedWeightsMomentum.getRow(rowIndex);
            final DataList momentumRow = weightsMomentum.getRow(rowIndex);
            final DataList gradientRow = gradientWRTWeights.getRow(rowIndex);

            for (int i = 0; i < updatedMomentumRow.getDataListSize(); ++i) {
                final double updatedValue =
                        this.beta1 * momentumRow.getValue(i) + (1.0 - this.beta1) * gradientRow.getValue(i);
                updatedMomentumRow.setValue(i, updatedValue);
            }
        }

        return updatedWeightsMomentum;
    }

    private Batch getCorrectedWeightsMomentum(final Batch weightsMomentum) {
        final int currentIteration = this.getCurrentIteration();
        final Batch correctedWeightsMomentum = new Batch();

        for (int rowIndex = 0; rowIndex < weightsMomentum.getRowsSize(); ++rowIndex) {
            final DataList weightsMomentumRow = weightsMomentum.getRow(rowIndex);
            final DataList updatedWeightsMomentumRow = new DataList(weightsMomentumRow.getDataListSize());

            for (int i = 0; i < weightsMomentumRow.getDataListSize(); ++i) {
                final double updatedValue =
                        weightsMomentumRow.getValue(i) / (1.0 - Math.pow(this.beta1, currentIteration + 1.0));
                updatedWeightsMomentumRow.setValue(i, updatedValue);
            }

            correctedWeightsMomentum.addRow(updatedWeightsMomentumRow);
        }

        return correctedWeightsMomentum;
    }

    private DataList getUpdatedBiasesCache(final HiddenLayer layer, final DataList biasesCache) {
        final DataList gradientWRTBiases = layer.getSavedOutputGradientStruct().getGradientWithRespectToBiases().getRow(0);
        final DataList updatedBiasesCache = new DataList(gradientWRTBiases.getDataListSize());

        for (int i = 0; i < gradientWRTBiases.getDataListSize(); ++i) {
            final double updatedValue =
                    this.beta2 * biasesCache.getValue(i) + (1.0 - this.beta2) * Math.pow(gradientWRTBiases.getValue(i), 2.0);
            updatedBiasesCache.setValue(i, updatedValue);
        }

        return updatedBiasesCache;
    }

    private DataList getCorrectedBiasesCache(final DataList biasesCache) {
        final int currentIteration = this.getCurrentIteration();
        final DataList correctedBiasesCache = new DataList(biasesCache.getDataListSize());

        for (int i = 0; i < biasesCache.getDataListSize(); ++i) {
            final double correctedValue =
                    biasesCache.getValue(i) / (1.0 - Math.pow(this.beta2, currentIteration + 1.0));
            correctedBiasesCache.setValue(i, correctedValue);
        }

        return correctedBiasesCache;
    }

    private Batch getUpdatedWeightsCache(final HiddenLayer layer, final Batch weightsCache) {
        final Batch gradientWRTWeights = layer.getSavedOutputGradientStruct().getGradientWithRespectToWeights();
        final Batch updatedWeightsCache = new Batch();

        for (int rowIndex = 0; rowIndex < gradientWRTWeights.getRowsSize(); ++rowIndex) {
            final DataList weightsCacheRow = weightsCache.getRow(rowIndex);
            final DataList gradientRow = gradientWRTWeights.getRow(rowIndex);
            final DataList updatedWeightsRow = new DataList(gradientRow.getDataListSize());

            for (int i = 0; i < gradientRow.getDataListSize(); ++i) {
                final double updatedValue =
                        this.beta2 * weightsCacheRow.getValue(i) + (1.0 - this.beta2) * Math.pow(gradientRow.getValue(i), 2.0);
                updatedWeightsRow.setValue(i, updatedValue);
            }

            updatedWeightsCache.addRow(updatedWeightsRow);
        }

        return updatedWeightsCache;
    }

    private Batch getCorrectedWeightsCache(final Batch weightsCache) {
        final int currentIteration = this.getCurrentIteration();
        final Batch correctedWeightsCache = new Batch();

        for (int rowIndex = 0; rowIndex < weightsCache.getRowsSize(); ++rowIndex) {
            final DataList weightsCacheRow = weightsCache.getRow(rowIndex);
            final DataList updatedWeightsCacheRow = new DataList(weightsCacheRow.getDataListSize());

            for (int i = 0; i < weightsCacheRow.getDataListSize(); ++i) {
                final double updatedValue =
                        weightsCacheRow.getValue(i) / (1.0 - Math.pow(this.beta2, currentIteration + 1.0));
                updatedWeightsCacheRow.setValue(i, updatedValue);
            }

            correctedWeightsCache.addRow(updatedWeightsCacheRow);
        }

        return correctedWeightsCache;
    }

    private void optimizeBiasesAndWeights(
            final HiddenLayer layer,
            final DataList correctedBiasesMomentum,
            final Batch correctedWeightsMomentum,
            final DataList correctedBiasesCache,
            final Batch correctedWeightsCache
    ) {
        final double currentLearningRate = this.getCurrentLearningRate();
        final List<Neuron> neurons = layer.getNeurons();

        for (int neuronIndex = 0; neuronIndex < layer.getNeuronsSize(); ++neuronIndex) {
            final Neuron neuron = neurons.get(neuronIndex);

            final double originalBias = neuron.getBias();
            final double updatedBias =
                    originalBias - (
                            currentLearningRate * correctedBiasesMomentum.getValue(neuronIndex)
                            / (Math.sqrt(correctedBiasesCache.getValue(neuronIndex)) + this.epsilon)
                    );
            neuron.setBias(updatedBias);

            final DataList neuronWeights = neuron.getWeights();
            final DataList neuronCorrectedWeightsMomentum = correctedWeightsMomentum.getColumn(neuronIndex);
            final DataList neuronCorrectedWeightsCache = correctedWeightsCache.getColumn(neuronIndex);

            for (int weightIndex = 0; weightIndex < neuronWeights.getDataListSize(); ++weightIndex) {
                final double originalWeight = neuronWeights.getValue(weightIndex);
                final double updatedWeight =
                        originalWeight - (
                                currentLearningRate * neuronCorrectedWeightsMomentum.getValue(weightIndex)
                                / (Math.sqrt(neuronCorrectedWeightsCache.getValue(weightIndex)) + this.epsilon)
                        );
                neuronWeights.setValue(weightIndex, updatedWeight);
            }
        }
    }
}
