package NeuralNetwork;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.BuildingBlocks.GradientStruct;
import NeuralNetwork.Layers.Common.ActivationLayer;
import NeuralNetwork.Layers.Common.HiddenLayer;
import NeuralNetwork.Layers.Common.LossLayer;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.Layers.Special.SoftmaxCategoricalCrossEntropyLayer;
import Utilities.CustomMath;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final int inputSize;
    private final List<LayerBase> layers;

    private boolean forwardStepExecuted;
    private boolean backwardStepExecuted;

    public NeuralNetwork(final int inputSize) {
        this.inputSize = inputSize;
        this.layers = new ArrayList<>();

        this.forwardStepExecuted = false;
        this.backwardStepExecuted = false;
    }

    public boolean isBackwardStepExecuted() {
        return this.backwardStepExecuted;
    }

    public List<LayerBase> getLayers() {
        return this.layers;
    }

    public void forward(final Batch inputBatch, final Batch targetBatch) {
        if (!this.isLastLayerSuitable()) {
            throw new RuntimeException("Cannot perform forward method, last layer is not suitable");
        }

        final LayerBase lastLayer = this.layers.getLast();
        lastLayer.setSavedTargetBatch(targetBatch);

        final HiddenLayer firstLayer = this.getLayerAsType(0, HiddenLayer.class);
        firstLayer.forward(inputBatch);

        for (int i = 1; i < this.layers.size(); ++i) {
            final LayerBase previousLayer = this.layers.get(i - 1);
            final LayerBase currentLayer = this.layers.get(i);

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
            if (!(lastLayer instanceof ActivationLayer)) {
                throw new IllegalArgumentException("Layer before hidden layer must be activation layer");
            }
            assert(this.layers.size() >= 2);

            final HiddenLayer penultimateLayer = this.getLayerAsType(this.layers.size() - 2, HiddenLayer.class);
            final int penultimateLayerNeuronsSize = penultimateLayer.getNeuronsSize();
            final int hiddenLayerToBeAddedWeightsSize = hiddenLayerToBeAdded.getWeightsSize();

            if (penultimateLayerNeuronsSize != hiddenLayerToBeAddedWeightsSize) {
                throw new IllegalArgumentException(
                        "Last hidden layer neurons size [" + penultimateLayerNeuronsSize + "] is not equal to new hidden layer weights size [" + hiddenLayerToBeAddedWeightsSize + "]"
                );
            }
        }

        this.layers.add(hiddenLayerToBeAdded);
    }

    public void addActivationLayer(final ActivationLayer activationLayerToBeAdded) {
        if (this.layers.isEmpty()) {
            throw new IllegalArgumentException("Activation layer cannot be the first layer in neural network");
        }

        final LayerBase lastLayer = this.layers.getLast();
        if (!(lastLayer instanceof HiddenLayer)) {
            throw new IllegalArgumentException("Layer before activation layer must be hidden layer");
        }

        this.layers.add(activationLayerToBeAdded);
    }

    public void addLossLayer(final LossLayer lossLayerToBeAdded) {
        if (this.layers.isEmpty()) {
            throw new IllegalArgumentException("Loss layer cannot be the first layer in neural network");
        }

        final LayerBase lastLayer = this.layers.getLast();
        if (!(lastLayer instanceof ActivationLayer)) {
            throw new IllegalArgumentException("Layer before loss layer must be activation layer");
        }

        this.layers.add(lossLayerToBeAdded);
    }

    public void addSpecialLayer(final LayerBase specialLayerToBeAdded) {
        if (specialLayerToBeAdded instanceof final SoftmaxCategoricalCrossEntropyLayer softmaxCCELayer) {
            if (this.layers.isEmpty()) {
                throw new IllegalArgumentException("Softmax categorical cross entropy layer cannot be the first layer in neural network");
            }

            final LayerBase lastLayer = this.layers.getLast();
            if (!(lastLayer instanceof HiddenLayer)) {
                throw new IllegalArgumentException("Layer before softmax categorical cross entropy layer must be hidden layer");
            }

            this.layers.add(specialLayerToBeAdded);
        } else {
            throw new IllegalArgumentException("Unknown special layer type");
        }
    }

    private <T extends LayerBase> T getLayerAsType(final int index, final Class<T> type) {
        final LayerBase layer = this.layers.get(index);

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
