package NeuralNetwork.Layers;

import NeuralNetwork.Batch;
import NeuralNetwork.DataRow;

public interface ILayer {
    default Batch calculateOutputBatch(final Batch batch) {
        final Batch outputBatch = new Batch();

        for (int i = 0; i < batch.getBatchSize(); ++i) {
            final DataRow inputRow = batch.getDataRow(i);
            final DataRow calculatedRow = this.calculateOutputRow(inputRow);

            outputBatch.addDataRow(calculatedRow);
        }

        return outputBatch;
    }

    DataRow calculateOutputRow(final DataRow inputRow);
}
