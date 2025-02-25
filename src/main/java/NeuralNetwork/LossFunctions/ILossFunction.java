package NeuralNetwork.LossFunctions;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;

public interface ILossFunction {
    default DataList calculate(final Batch predictedBatch, final Batch targetBatch) {
        final DataList outputRow = new DataList(predictedBatch.getRowsSize());

        for (int i = 0; i < predictedBatch.getRowsSize(); ++i) {
            final DataList predictedRow = predictedBatch.getRow(i);
            final DataList targetRow = targetBatch.getRow(i);

            final double loss = this.calculate(predictedRow, targetRow);
            outputRow.setValue(i, loss);
        }

        return outputRow;
    }

    default DataList calculate(final Batch predictedBatch, final Integer[] targetIndexes) {
        final DataList outputRow = new DataList(predictedBatch.getRowsSize());

        for (int i = 0; i < predictedBatch.getRowsSize(); ++i) {
            final DataList predictedRow = predictedBatch.getRow(i);
            final int targetIndex = targetIndexes[i];

            final double loss = this.calculate(predictedRow, targetIndex);
            outputRow.setValue(i, loss);
        }

        return outputRow;
    }

    double calculate(final DataList predictedRow, final DataList targetRow);
    double calculate(final DataList predictedRow, final int targetIndex);
}
