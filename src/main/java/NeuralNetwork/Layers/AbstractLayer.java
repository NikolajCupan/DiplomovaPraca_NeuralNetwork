package NeuralNetwork.Layers;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;

import java.util.Optional;

public abstract class AbstractLayer {
    private Optional<Batch> processedInputBatch;

    protected AbstractLayer() {
        this.processedInputBatch = Optional.empty();
    }

    public Batch calculateOutputBatch(final Batch inputBatch) {
        if (this.processedInputBatch.isPresent()) {
            throw new IllegalArgumentException("Layer already processed an input batch");
        }
        this.processedInputBatch = Optional.of(inputBatch);

        final Batch outputBatch = new Batch();

        for (int i = 0; i < inputBatch.getRowsSize(); ++i) {
            final DataList inputRow = inputBatch.getRow(i);
            final DataList calculatedRow = this.calculateOutputRow(inputRow);

            outputBatch.addRow(calculatedRow);
        }

        return outputBatch;
    }

    public DataList getInputs(final int inputIndex) {
        if (this.processedInputBatch.isEmpty()) {
            throw new IllegalArgumentException("Layer has not processed an input batch yet");
        }

        final int batchSize = this.processedInputBatch.get().getRowsSize();
        final DataList inputs = new DataList(batchSize);

        for (int i = 0; i < batchSize; ++i) {
            final DataList inputRow = this.processedInputBatch.get().getRow(i);
            inputs.setValue(i, inputRow.getValue(inputIndex));
        }

        return inputs;
    }

    public int getInputsRowSize() {
        if (this.processedInputBatch.isEmpty()) {
            throw new IllegalArgumentException("Layer has not processed an input batch yet");
        }

        return this.processedInputBatch.get().getRow(0).getDataListSize();
    }

    protected abstract DataList calculateOutputRow(final DataList inputRow);
}
