package NeuralNetwork.LossFunctions;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import Utilities.CustomMath;

public class MeanAbsoluteError implements IRegressionLossFunction {
    @Override
    public Batch backward(final Batch inputBatch, final Batch targetBatch) {
        final int inputBatchRowsSize = inputBatch.getRowsSize();
        final int inputBatchColumnsSize = inputBatch.getColumnsSize();

        final Batch gradientWRTInputs = new Batch();

        for (int rowIndex = 0; rowIndex < inputBatchRowsSize; ++rowIndex) {
            final DataList inputBatchRow = inputBatch.getRow(rowIndex);
            final DataList targetBatchRow = targetBatch.getRow(rowIndex);

            final DataList outputRow = new DataList(inputBatchColumnsSize);

            for (int columnIndex = 0; columnIndex < inputBatchColumnsSize; ++columnIndex) {
                final double predictedValue = inputBatchRow.getValue(columnIndex);
                final double targetValue = targetBatchRow.getValue(columnIndex);

                final double sign = Math.signum(targetValue - predictedValue);
                final double normalizedValue =
                        (sign / inputBatchColumnsSize) / inputBatchRowsSize;

                outputRow.setValue(columnIndex, normalizedValue);
            }

            gradientWRTInputs.addRow(outputRow);
        }

        return gradientWRTInputs;
    }

    @Override
    public double loss(final DataList predictedRow, final DataList targetRow) {
        if (predictedRow.getDataListSize() != targetRow.getDataListSize()) {
            throw new IllegalArgumentException("Predicted row size [" + predictedRow.getDataListSize() + "] is not equal to target row size [" + targetRow.getDataListSize() + "]");
        }

        final DataList losses = new DataList(predictedRow.getDataListSize());

        for (int i = 0; i < predictedRow.getDataListSize(); ++i) {
            final double predictedValue = predictedRow.getValue(i);
            final double targetValue = targetRow.getValue(i);

            final double loss = Math.abs(targetValue - predictedValue);
            losses.setValue(i, loss);
        }

        return CustomMath.mean(losses);
    }

    @Override
    public String toString() {
        return "Mean absolute error";
    }
}
