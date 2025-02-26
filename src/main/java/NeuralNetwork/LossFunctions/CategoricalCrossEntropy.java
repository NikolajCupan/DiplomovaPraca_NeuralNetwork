package NeuralNetwork.LossFunctions;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import Utilities.CustomMath;

public class CategoricalCrossEntropy extends AbstractLossFunction {
    private static final double CLAMP_VALUE = 0.0000001;

    @Override
    public Batch backward(final Batch predictedBatch, final Batch targetBatch) {
        final int predictedBatchRowsSize = predictedBatch.getRowsSize();
        final int predictedBatchColumnsSize = predictedBatch.getColumnsSize();

        final Batch outputBatch = new Batch();

        for (int rowIndex = 0; rowIndex < predictedBatchRowsSize; ++rowIndex) {
            final DataList predictedBatchRow = predictedBatch.getRow(rowIndex);
            final DataList targetBatchRow = targetBatch.getRow(rowIndex);

            final DataList outputRow = new DataList(predictedBatchColumnsSize);

            for (int columnIndex = 0; columnIndex < predictedBatchColumnsSize; ++columnIndex) {
                final double value = -targetBatchRow.getValue(columnIndex) / predictedBatchRow.getValue(columnIndex);
                final double normalizedValue = value / predictedBatchRowsSize;

                outputRow.setValue(columnIndex, normalizedValue);
            }

            outputBatch.addRow(outputRow);
        }

        return outputBatch;
    }

    @Override
    protected double calculate(final DataList predictedList, final DataList targetList) {
        final int targetListArgMax = CustomMath.argMax(targetList);

        final double targetListMax = targetList.getValue(targetListArgMax);
        if (Math.abs(1.0 - targetListMax) > 0.0) {
            throw new IllegalArgumentException("Categorical cross entropy requires one-hot target list row");
        }

        return this.calculate(predictedList, targetListArgMax);
    }

    @Override
    protected double calculate(final DataList predictedList, final int targetIndex) {
        final double predictedValue = predictedList.getValue(targetIndex);
        final double clampedValue = Math.clamp(predictedValue, CLAMP_VALUE, 1.0 - CLAMP_VALUE);

        return -Math.log(clampedValue);
    }
}
