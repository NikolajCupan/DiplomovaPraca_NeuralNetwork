package NeuralNetwork.Layers;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;

import java.util.Optional;

public abstract class AbstractLayer {
    private Optional<Batch> savedInputBatch;

    protected AbstractLayer() {
        this.savedInputBatch = Optional.empty();
    }

    public Batch forward(final Batch inputBatch) {
        if (this.savedInputBatch.isPresent()) {
            throw new IllegalArgumentException("Layer already processed an input batch");
        }
        this.savedInputBatch = Optional.of(inputBatch);

        final Batch outputBatch = new Batch();

        for (int rowIndex = 0; rowIndex < inputBatch.getRowsSize(); ++rowIndex) {
            final DataList inputRow = inputBatch.getRow(rowIndex);

            final DataList calculatedRow = this.forward(inputRow);
            outputBatch.addRow(calculatedRow);
        }

        return outputBatch;
    }

    protected Batch getSavedInputBatch() {
        if (this.savedInputBatch.isEmpty()) {
            throw new IllegalArgumentException("Layer has not processed an input batch yet");
        }

        return this.savedInputBatch.get();
    }

    public abstract Batch calculateGradientWithRespectToBiases(final Batch inputGradientBatch);
    public abstract Batch calculateGradientWithRespectToWeights(final Batch inputGradientBatch);
    public abstract Batch calculateGradientWithRespectToInputs(final Batch inputGradientBatch);
    protected abstract DataList calculateGradientWithRespectToInputs(final DataList inputGradient);

    protected abstract DataList forward(final DataList inputRow);
}
