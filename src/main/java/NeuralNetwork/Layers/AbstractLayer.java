package NeuralNetwork.Layers;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;

import java.util.Optional;

public abstract class AbstractLayer {
    private Optional<Batch> savedInputBatch;

    protected AbstractLayer() {
        this.savedInputBatch = Optional.empty();
    }

    public Batch calculateOutputBatch(final Batch inputBatch) {
        if (this.savedInputBatch.isPresent()) {
            throw new IllegalArgumentException("Layer already processed an input batch");
        }
        this.savedInputBatch = Optional.of(inputBatch);

        final Batch outputBatch = new Batch();

        for (int i = 0; i < inputBatch.getRowsSize(); ++i) {
            final DataList inputRow = inputBatch.getRow(i);
            final DataList calculatedRow = this.calculateOutputRow(inputRow);

            outputBatch.addRow(calculatedRow);
        }

        return outputBatch;
    }

    protected Batch getInputBatch() {
        if (this.savedInputBatch.isEmpty()) {
            throw new IllegalArgumentException("Layer has not processed an input batch yet");
        }

        return this.savedInputBatch.get();
    }

    public abstract Batch calculateGradientWithRespectToInputs(final Batch inputGradientBatch);
    protected abstract DataList calculateGradientWithRespectToInputs(final DataList inputGradient);

    protected abstract DataList calculateOutputRow(final DataList inputRow);
}
