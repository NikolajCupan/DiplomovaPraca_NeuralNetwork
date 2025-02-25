package NeuralNetwork.LossFunctions;

import NeuralNetwork.Batch;
import NeuralNetwork.DataRow;

public interface ILossFunction {
    default DataRow calculate(final Batch predictedBatch, final Batch targetBatch) {
        final DataRow outputRow = new DataRow(predictedBatch.getBatchSize());

        for (int i = 0; i < predictedBatch.getBatchSize(); ++i) {
            final DataRow predictedRow = predictedBatch.getDataRow(i);
            final DataRow targetRow = targetBatch.getDataRow(i);

            final double loss = this.calculate(predictedRow, targetRow);
            outputRow.setValue(i, loss);
        }

        return outputRow;
    }

    default DataRow calculate(final Batch predictedBatch, final Integer[] targetIndexes) {
        final DataRow outputRow = new DataRow(predictedBatch.getBatchSize());

        for (int i = 0; i < predictedBatch.getBatchSize(); ++i) {
            final DataRow predictedRow = predictedBatch.getDataRow(i);
            final int targetIndex = targetIndexes[i];

            final double loss = this.calculate(predictedRow, targetIndex);
            outputRow.setValue(i, loss);
        }

        return outputRow;
    }

    double calculate(final DataRow predictedRow, final DataRow targetRow);
    double calculate(final DataRow predictedRow, final int targetIndex);
}
