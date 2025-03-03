package NeuralNetwork;

import NeuralNetwork.BuildingBlocks.*;
import NeuralNetwork.Layers.Common.ActivationLayer;
import NeuralNetwork.Layers.Common.DropoutLayer;
import NeuralNetwork.Layers.Common.HiddenLayer;
import NeuralNetwork.Layers.Common.LossLayer;
import NeuralNetwork.Layers.IAccuracyLayerBase;
import NeuralNetwork.Layers.ILossLayerBase;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.Layers.Special.SoftmaxCategoricalCrossEntropyLayer;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class NeuralNetwork {
    private final int inputSize;
    private final List<LayerBase> layers;

    private boolean forwardStepExecuted;
    private boolean backwardStepExecuted;

    private Optional<RegularizerStruct> globalRegularizerStruct;

    public NeuralNetwork(final int inputSize) {
        this.inputSize = inputSize;
        this.layers = new ArrayList<>();

        this.forwardStepExecuted = false;
        this.backwardStepExecuted = false;

        this.globalRegularizerStruct = Optional.empty();
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

    public double getAccuracy() {
        if (!this.forwardStepExecuted) {
            throw new RuntimeException("Cannot calculate accuracy before forward step");
        }

        final LayerBase lastLayer = this.layers.getLast();

        if (lastLayer instanceof final IAccuracyLayerBase lossLayer) {
            return lossLayer.getAccuracy();
        } else {
            throw new RuntimeException("Last layer cannot be used to calculate accuracy");
        }
    }

    public double getLoss() {
        if (!this.forwardStepExecuted) {
            throw new RuntimeException("Cannot calculate loss before forward step");
        }

        final LayerBase lastLayer = this.layers.getLast();

        if (lastLayer instanceof final ILossLayerBase lossLayer) {
            return lossLayer.getLoss();
        } else {
            throw new RuntimeException("Last layer cannot be used to calculate loss");
        }
    }

    public double getRegularizedLoss() {
        double regularizedLoss = this.getLoss();

        for (final LayerBase layer : this.layers) {
            if (layer instanceof final HiddenLayer hiddenLayer && hiddenLayer.isRegularizerPresent()) {
                regularizedLoss += hiddenLayer.getRegularizedLoss();
            }
        }

        return regularizedLoss;
    }

    public boolean isBackwardStepExecuted() {
        return this.backwardStepExecuted;
    }

    public List<LayerBase> getLayers() {
        return this.layers;
    }

    public void forward(final Batch inputBatch, final Batch targetBatch, final boolean excludeDropoutLayers) {
        if (!this.isLastLayerSuitable()) {
            throw new RuntimeException("Cannot perform forward method, last layer is not suitable");
        }

        final List<LayerBase> usedLayers = new ArrayList<>();
        for (final LayerBase layer : this.layers) {
            if (!excludeDropoutLayers || !(layer instanceof DropoutLayer)) {
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

        this.forwardStepExecuted = true;
    }

    public void backward() {
        if (!this.forwardStepExecuted) {
            throw new RuntimeException("Cannot perform backward step before forward step");
        }

        final LayerBase lastLayer = this.layers.getLast();
        lastLayer.backward(new GradientStruct());

        for (int i = this.layers.size() - 1; i > 0; --i) {
            final LayerBase previousLayer = this.layers.get(i - 1);
            final LayerBase currentLayer = this.layers.get(i);

            previousLayer.backward(currentLayer.getSavedOutputGradientStruct());
        }

        this.backwardStepExecuted = true;
    }

    public void clearState() {
        if (!this.backwardStepExecuted) {
            throw new RuntimeException("Cannot clear state before backward step");
        }

        this.forwardStepExecuted = false;
        this.backwardStepExecuted = false;

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
