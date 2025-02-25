package NeuralNetwork.Layers;

import NeuralNetwork.Batch;
import NeuralNetwork.DataRow;

public interface ILayer {
    default Batch calculateOutputBatch(final Batch inputBatch) {
        final Batch outputBatch = new Batch();

        for (int i = 0; i < inputBatch.getBatchSize(); ++i) {
            final DataRow inputRow = inputBatch.getDataRow(i);
            final DataRow calculatedRow = this.calculateOutputRow(inputRow);

            outputBatch.addDataRow(calculatedRow);
        }

        return outputBatch;
    }

    DataRow calculateOutputRow(final DataRow inputRow);
}
