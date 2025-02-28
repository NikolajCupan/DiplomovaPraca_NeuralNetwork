package NeuralNetwork.LossFunctions;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import Utilities.CustomMath;

public class CategoricalCrossEntropy implements ILossFunction {
    private static final double CLAMP_VALUE = 0.0000001;

    public Batch backward(final Batch predictedBatch, final Batch targetBatch) {
        final int predictedBatchRowsSize = predictedBatch.getRowsSize();
        final int predictedBatchColumnsSize = predictedBatch.getColumnsSize();

        final Batch gradientWRTInputs = new Batch();

        for (int rowIndex = 0; rowIndex < predictedBatchRowsSize; ++rowIndex) {
            final DataList predictedBatchRow = predictedBatch.getRow(rowIndex);
            final DataList targetBatchRow = targetBatch.getRow(rowIndex);

            final DataList outputRow = new DataList(predictedBatchColumnsSize);

            for (int columnIndex = 0; columnIndex < predictedBatchColumnsSize; ++columnIndex) {
                final double value = -targetBatchRow.getValue(columnIndex) / predictedBatchRow.getValue(columnIndex);
                final double normalizedValue = value / predictedBatchRowsSize;

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

        final int targetRowArgMax = CustomMath.argMax(targetRow);

        final double targetRowMax = targetRow.getValue(targetRowArgMax);
        if (Math.abs(1.0 - targetRowMax) > 0.0) {
            throw new IllegalArgumentException("Categorical cross entropy requires one-hot target row");
        }

        final double predictedValue = predictedRow.getValue(targetRowArgMax);
        final double clampedValue = Math.clamp(predictedValue, CLAMP_VALUE, 1.0 - CLAMP_VALUE);
        return -Math.log(clampedValue);
    }
}
