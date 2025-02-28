package NeuralNetwork.Layers;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.GradientStruct;

import java.util.Optional;

public abstract class LayerBase {
    private Optional<Batch> savedInputBatch;
    private Optional<Batch> savedOutputBatch;
    private Optional<Batch> savedTargetBatch;

    private Optional<GradientStruct> savedInputGradientStruct;
    private Optional<GradientStruct> savedOutputGradientStruct;

    protected LayerBase() {
        this.savedInputBatch = Optional.empty();
        this.savedOutputBatch = Optional.empty();
        this.savedTargetBatch = Optional.empty();

        this.savedInputGradientStruct = Optional.empty();
        this.savedOutputGradientStruct = Optional.empty();
    }

    public void clearState() {
        this.savedInputBatch = Optional.empty();
        this.savedOutputBatch = Optional.empty();
        this.savedTargetBatch = Optional.empty();

        this.savedInputGradientStruct = Optional.empty();
        this.savedOutputGradientStruct = Optional.empty();
    }

    public Batch getSavedInputBatch() {
        if (this.savedInputBatch.isEmpty()) {
            throw new IllegalArgumentException("Layer saved input batch is empty");
        }

        return this.savedInputBatch.get();
    }

    public Batch getSavedOutputBatch() {
        if (this.savedOutputBatch.isEmpty()) {
            throw new IllegalArgumentException("Layer saved output batch is empty");
        }

        return this.savedOutputBatch.get();
    }

    public Batch getSavedTargetBatch() {
        if (this.savedTargetBatch.isEmpty()) {
            throw new IllegalArgumentException("Layer saved target batch is empty");
        }

        return this.savedTargetBatch.get();
    }

    public GradientStruct getSavedOutputGradientStruct() {
        if (this.savedOutputGradientStruct.isEmpty()) {
            throw new IllegalArgumentException("Layer saved output gradient struct is empty");
        }

        return this.savedOutputGradientStruct.get();
    }

    public void setSavedInputBatch(final Batch inputBatch) {
        if (this.savedInputBatch.isPresent()) {
            throw new IllegalArgumentException("Layer base already has input batch set");
        }

        this.savedInputBatch = Optional.of(inputBatch);
    }

    public void setSavedOutputBatch(final Batch outputBatch) {
        if (this.savedOutputBatch.isPresent()) {
            throw new IllegalArgumentException("Layer base already has output batch set");
        }

        this.savedOutputBatch = Optional.of(outputBatch);
    }

    public void setSavedTargetBatch(final Batch targetBatch) {
        if (this.savedTargetBatch.isPresent()) {
            throw new IllegalArgumentException("Layer base already has target batch set");
        }

        this.savedTargetBatch = Optional.of(targetBatch);
    }

    public void setSavedInputGradientStruct(final GradientStruct inputGradientStruct) {
        if (this.savedInputGradientStruct.isPresent()) {
            throw new IllegalArgumentException("Layer base already has input gradient struct set");
        }

        this.savedInputGradientStruct = Optional.of(inputGradientStruct);
    }

    public void setSavedOutputGradientStruct(final GradientStruct outputGradientStruct) {
        if (this.savedOutputGradientStruct.isPresent()) {
            throw new IllegalArgumentException("Layer base already has output gradient struct set");
        }

        this.savedOutputGradientStruct = Optional.of(outputGradientStruct);
    }

    public abstract void forward(final Batch inputBatch);
    public abstract void backward(final GradientStruct inputGradientStruct);
}
