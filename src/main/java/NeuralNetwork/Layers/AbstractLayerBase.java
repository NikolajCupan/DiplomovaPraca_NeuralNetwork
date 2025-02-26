package NeuralNetwork.Layers;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import Utilities.GradientStruct;

import java.util.Optional;

public abstract class AbstractLayerBase {
    private Optional<Batch> savedInputBatch;
    private Optional<Batch> savedOutputBatch;

    protected AbstractLayerBase() {
        this.savedInputBatch = Optional.empty();
        this.savedOutputBatch = Optional.empty();
    }

    protected Batch getSavedInputBatch() {
        if (this.savedInputBatch.isEmpty()) {
            throw new IllegalArgumentException("Layer has not processed an input batch yet");
        }

        return this.savedInputBatch.get();
    }

    protected Batch getSavedOutputBatch() {
        if (this.savedOutputBatch.isEmpty()) {
            throw new IllegalArgumentException("Layer has not processed an input batch yet");
        }

        return this.savedOutputBatch.get();
    }

    public Batch forward(final Batch inputBatch) {
        if (this.savedInputBatch.isPresent() || this.savedOutputBatch.isPresent()) {
            throw new IllegalArgumentException("Layer already processed an input batch");
        }
        this.savedInputBatch = Optional.of(inputBatch);

        final Batch outputBatch = new Batch();

        for (int rowIndex = 0; rowIndex < inputBatch.getRowsSize(); ++rowIndex) {
            final DataList inputRow = inputBatch.getRow(rowIndex);

            final DataList calculatedRow = this.forward(inputRow);
            outputBatch.addRow(calculatedRow);
        }

        this.savedOutputBatch = Optional.of(outputBatch);
        return outputBatch;
    }

    protected abstract DataList forward(final DataList inputRow);
    public abstract GradientStruct backward(final Batch inputGradientBatch);
}
