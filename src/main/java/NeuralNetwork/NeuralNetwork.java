package NeuralNetwork;

import NeuralNetwork.BuildingBlocks.*;
import NeuralNetwork.Layers.Common.ActivationLayer;
import NeuralNetwork.Layers.Common.DropoutLayer;
import NeuralNetwork.Layers.Common.HiddenLayer;
import NeuralNetwork.Layers.Common.LossLayer;
import NeuralNetwork.Layers.IAccuracyForPrintingGetter;
import NeuralNetwork.Layers.ILossForPrintingGetter;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.Layers.Special.SoftmaxCategoricalCrossEntropyLayer;
import NeuralNetwork.Optimizers.OptimizerBase;
import Utilities.Helper;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class NeuralNetwork {
    private final int inputSize;
    private final List<LayerBase> layers;

    private Optional<RegularizerStruct> globalRegularizerStruct;
    private Optional<OptimizerBase> optimizer;

    public NeuralNetwork(final int inputSize) {
        this.inputSize = inputSize;
        this.layers = new ArrayList<>();

        this.globalRegularizerStruct = Optional.empty();
        this.optimizer = Optional.empty();
    }

    public void initializeGlobalRegularizer(
            final double biasesRegularizerL1,
            final double biasesRegularizerL2,
            final double weightsRegularizerL1,
            final double weightsRegularizerL2) {
        if (this.globalRegularizerStruct.isPresent()) {
            throw new RuntimeException("Global regularizer is already set in neural network");
        }

        if (!this.layers.isEmpty()) {
            throw new RuntimeException("Global regularizer cannot be set if layers are not empty");
        }

        final RegularizerStruct regularizer = new RegularizerStruct();
        regularizer.setBiasesRegularizerL1(biasesRegularizerL1);
        regularizer.setBiasesRegularizerL2(biasesRegularizerL2);
        regularizer.setWeightsRegularizerL1(weightsRegularizerL1);
        regularizer.setWeightsRegularizerL2(weightsRegularizerL2);

        this.globalRegularizerStruct = Optional.of(regularizer);
    }

    private double getAccuracyForPrinting() {
        final LayerBase lastLayer = this.layers.getLast();

        if (lastLayer instanceof final IAccuracyForPrintingGetter lossLayer) {
            return lossLayer.getAccuracyForPrinting();
        } else {
            throw new RuntimeException("Last layer cannot be used to calculate accuracy");
        }
    }

    private double getLossForPrinting() {
        final LayerBase lastLayer = this.layers.getLast();

        if (lastLayer instanceof final ILossForPrintingGetter lossLayer) {
            return lossLayer.getLossForPrinting();
        } else {
            throw new RuntimeException("Last layer cannot be used to calculate loss");
        }
    }

    private double getRegularizedLossForPrinting() {
        double regularizedLoss = this.getLossForPrinting();

        for (final LayerBase layer : this.layers) {
            if (layer instanceof final HiddenLayer hiddenLayer && hiddenLayer.isRegularizerPresent()) {
                regularizedLoss += hiddenLayer.getRegularizedLoss();
            }
        }

        return regularizedLoss;
    }

    public List<LayerBase> getLayers() {
        return this.layers;
    }

    public void train(final Batch inputBatch, final Batch targetBatch, final int epochsSize, final int printStep) {
        if (!this.isLastLayerSuitable()) {
            throw new RuntimeException("Cannot perform forward method, last layer is not suitable");
        } else if (this.optimizer.isEmpty()) {
            throw new RuntimeException("Cannot perform training, optimizer is empty");
        }

        for (int epoch = 1; epoch < epochsSize + 1; ++epoch) {
            boolean printing = (epoch % printStep == 0);

            this.forward(inputBatch, targetBatch, true);

            if (printing) {
                final double accuracy = this.getAccuracyForPrinting();
                final double loss = this.getLossForPrinting();
                final double regularizedLoss = this.getRegularizedLossForPrinting();

                System.out.printf(
                        "epoch: %-15d accuracy: %-15s loss: %-15s regularized loss: %-15s",
                        epoch,
                        Helper.formatNumber(accuracy, 5),
                        Helper.formatNumber(loss, 5),
                        Helper.formatNumber(regularizedLoss, 5)
                );
            }

            this.backward();
            this.optimizer.get().performOptimization();
            this.clearState();

            if (printing) {
                final double learningRate = this.optimizer.get().getCurrentLearningRate();
                System.out.printf("%-15s%n", "lr: " + learningRate);
            }
        }
    }

    public void test(final Batch inputBatch, final Batch targetBatch) {
        this.forward(inputBatch, targetBatch, false);

        final double accuracy = this.getAccuracyForPrinting();
        final double loss = this.getLossForPrinting();

        System.out.printf(
                "accuracy: %-15s loss: %-15s",
                Helper.formatNumber(accuracy, 5),
                Helper.formatNumber(loss, 5)
        );
    }

    private void forward(final Batch inputBatch, final Batch targetBatch, final boolean includeDropoutLayers) {
        final List<LayerBase> usedLayers = new ArrayList<>();
        for (final LayerBase layer : this.layers) {
            if (includeDropoutLayers || !(layer instanceof DropoutLayer)) {
                usedLayers.add(layer);
            }
        }

        final LayerBase lastLayer = usedLayers.getLast();
        lastLayer.setSavedTargetBatch(targetBatch);

        final HiddenLayer firstLayer = NeuralNetwork.getLayerAsType(usedLayers, 0, HiddenLayer.class);
        firstLayer.forward(inputBatch);

        for (int i = 1; i < usedLayers.size(); ++i) {
            final LayerBase previousLayer = usedLayers.get(i - 1);
            final LayerBase currentLayer = usedLayers.get(i);

            currentLayer.forward(previousLayer.getSavedOutputBatch());
        }
    }

    private void backward() {
        final LayerBase lastLayer = this.layers.getLast();
        lastLayer.backward(new GradientStruct());

        for (int i = this.layers.size() - 1; i > 0; --i) {
            final LayerBase previousLayer = this.layers.get(i - 1);
            final LayerBase currentLayer = this.layers.get(i);

            previousLayer.backward(currentLayer.getSavedOutputGradientStruct());
        }
    }

    private void clearState() {
        for (final LayerBase layer : this.layers) {
            layer.clearState();
        }
    }

    public void addHiddenLayer(final HiddenLayer hiddenLayerToBeAdded) {
        if (this.layers.isEmpty()) {
            final int firstLayerWeightsSize = hiddenLayerToBeAdded.getWeightsSize();

            if (this.inputSize != firstLayerWeightsSize) {
                throw new IllegalArgumentException("Input size [" + this.inputSize + "] is not equal to weights size of first layer [" + firstLayerWeightsSize + "]");
            }
        } else {
            final LayerBase lastLayer = this.layers.getLast();
            if (!hiddenLayerToBeAdded.isCompatible(lastLayer)) {
                throw new IllegalArgumentException("Hidden layer cannot be placed after " + lastLayer.getClass());
            }
            assert(this.layers.size() >= 2);

            final LayerBase penultimateLayer = this.layers.get(this.layers.size() - 2);

            if (penultimateLayer instanceof final HiddenLayer penultimateHiddenLayer) {
                final int penultimateLayerNeuronsSize = penultimateHiddenLayer.getNeuronsSize();
                final int hiddenLayerToBeAddedWeightsSize = hiddenLayerToBeAdded.getWeightsSize();

                if (penultimateLayerNeuronsSize != hiddenLayerToBeAddedWeightsSize) {
                    throw new IllegalArgumentException(
                            "Current last hidden layer neurons size [" + penultimateLayerNeuronsSize + "] is not equal to new hidden layer weights size [" + hiddenLayerToBeAddedWeightsSize + "]"
                    );
                }
            }
        }

        if (this.globalRegularizerStruct.isPresent()) {
            try {
                hiddenLayerToBeAdded.initializeRegularizer(this.globalRegularizerStruct.get());
            } catch (final Exception IllegalArgumentException) {
                throw new IllegalArgumentException("Neural network could not set regularizer for hidden layer, hidden layer already has regularizer set");
            }
        }

        this.layers.add(hiddenLayerToBeAdded);
    }

    public void addActivationLayer(final ActivationLayer activationLayerToBeAdded) {
        if (this.layers.isEmpty()) {
            throw new IllegalArgumentException("Activation layer cannot be the first layer in neural network");
        }

        final LayerBase lastLayer = this.layers.getLast();
        if (!activationLayerToBeAdded.isCompatible(lastLayer)) {
            throw new IllegalArgumentException("Activation layer cannot be placed after " + lastLayer.getClass());
        }

        this.layers.add(activationLayerToBeAdded);
    }

    public void addDropoutLayer(final DropoutLayer dropoutLayerToBeAdded) {
        if (this.layers.isEmpty()) {
            throw new IllegalArgumentException("Dropout layer cannot be the first layer in neural network");
        }

        final LayerBase lastLayer = this.layers.getLast();
        if (!dropoutLayerToBeAdded.isCompatible(lastLayer)) {
            throw new IllegalArgumentException("Dropout layer cannot be placed after " + lastLayer.getClass());
        }

        this.layers.add(dropoutLayerToBeAdded);
    }

    public void addLossLayer(final LossLayer lossLayerToBeAdded) {
        if (this.layers.isEmpty()) {
            throw new IllegalArgumentException("Loss layer cannot be the first layer in neural network");
        }

        final LayerBase lastLayer = this.layers.getLast();
        if (!lossLayerToBeAdded.isCompatible(lastLayer)) {
            throw new IllegalArgumentException("Loss layer cannot be placed after "  + lastLayer.getClass());
        }

        this.layers.add(lossLayerToBeAdded);
    }

    public void addSpecialLayer(final LayerBase specialLayerToBeAdded) {
        if (specialLayerToBeAdded instanceof final SoftmaxCategoricalCrossEntropyLayer softmaxCCELayer) {
            if (this.layers.isEmpty()) {
                throw new IllegalArgumentException("Softmax categorical cross entropy layer cannot be the first layer in neural network");
            }

            final LayerBase lastLayer = this.layers.getLast();
            if (!softmaxCCELayer.isCompatible(lastLayer)) {
                throw new IllegalArgumentException("Softmax categorical cross entropy layer cannot be placed after " + lastLayer.getClass());
            }

            this.layers.add(specialLayerToBeAdded);
        } else {
            throw new IllegalArgumentException("Unknown special layer type");
        }
    }

    public void setOptimizer(final OptimizerBase optimizer) {
        if (!this.isLastLayerSuitable()) {
            throw new IllegalArgumentException("Optimizer must be set when all layers are already present");
        } else if (this.optimizer.isPresent()) {
            throw new IllegalArgumentException("Optimizer is already set");
        }

        this.optimizer = Optional.of(optimizer);
    }

    private static <T extends LayerBase> T getLayerAsType(final List<LayerBase> layers, final int index, final Class<T> type) {
        final LayerBase layer = layers.get(index);

        if (type.isInstance(layer)) {
            return type.cast(layer);
        } else {
            throw new IllegalArgumentException("Layer at index [" + index + "] is not of type [" + type.getName() + "]");
        }
    }

    private boolean isLastLayerSuitable() {
        if (this.layers.isEmpty()) {
            return false;
        }

        final LayerBase lastLayer = this.layers.getLast();

        if (lastLayer instanceof LossLayer) {
            return true;
        } else if (lastLayer instanceof SoftmaxCategoricalCrossEntropyLayer) {
            return true;
        }

        return false;
    }
}
