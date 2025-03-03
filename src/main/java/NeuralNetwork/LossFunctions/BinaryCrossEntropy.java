package NeuralNetwork.LossFunctions;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import Utilities.CustomMath;

public class BinaryCrossEntropy implements ILossFunction {
    private static final double CLAMP_VALUE = 0.0000001;

    // "inputBatch" can be for example output from Sigmoid activation layer,
    // that means in loss layer it is an input
    public Batch backward(final Batch inputBatch, final Batch targetBatch) {
        final int inputBatchRowsSize = inputBatch.getRowsSize();
        final int inputBatchColumnsSize = inputBatch.getColumnsSize();

        final Batch gradientWRTInputs = new Batch();

        for (int rowIndex = 0; rowIndex < inputBatchRowsSize; ++rowIndex) {
            final DataList inputBatchRow = BinaryCrossEntropy.clampDataList(inputBatch.getRow(rowIndex));
            final DataList targetBatchRow = targetBatch.getRow(rowIndex);

            final DataList outputRow = new DataList(inputBatchColumnsSize);

            for (int columnIndex = 0; columnIndex < inputBatchColumnsSize; ++columnIndex) {
                final double predictedValue = inputBatchRow.getValue(columnIndex);
                final double targetValue = targetBatchRow.getValue(columnIndex);

                final double calculatedValue =
                        -(targetValue / predictedValue - (1 - targetValue) / (1 - predictedValue)) / inputBatchColumnsSize;
                outputRow.setValue(columnIndex, calculatedValue);
            }

            gradientWRTInputs.addRow(outputRow);
        }

        return gradientWRTInputs;
    }

    public double getAccuracy(final Batch predictedBatch, final Batch targetBatch) {
        assert(predictedBatch.getRowsSize() == targetBatch.getRowsSize());

        final DataList correctPredictionsList = new DataList(predictedBatch.getRowsSize());

        for (int rowIndex = 0; rowIndex < predictedBatch.getRowsSize(); ++rowIndex) {
            final DataList predictedRow = predictedBatch.getRow(rowIndex);
            final DataList targetRow = targetBatch.getRow(rowIndex);

            double correctPredictions = 0.0;

            for (int i = 0; i < predictedRow.getDataListSize(); ++i) {
                final int prediction = predictedRow.getValue(i) < 0.5 ? 0 : 1;
                final int target = targetRow.getValue(i) < 0.5 ? 0 : 1;

                if (prediction == target) {
                    correctPredictions += 1.0;
                }
            }

            correctPredictionsList.setValue(rowIndex, correctPredictions / predictedRow.getDataListSize());
        }

        return CustomMath.mean(correctPredictionsList);
    }

    public double getLoss(final Batch savedOutputBatch) {
        final DataList savedOutput = savedOutputBatch.getRow(0);
        return CustomMath.mean(savedOutput);
    }

    @Override
    public double loss(final DataList predictedRow, final DataList targetRow) {
        if (predictedRow.getDataListSize() != targetRow.getDataListSize()) {
            throw new IllegalArgumentException("Predicted row size [" + predictedRow.getDataListSize() + "] is not equal to target row size [" + targetRow.getDataListSize() + "]");
        }

        final DataList clampedPredictedRow = BinaryCrossEntropy.clampDataList(predictedRow);
        final DataList losses = new DataList(predictedRow.getDataListSize());

        for (int i = 0; i < predictedRow.getDataListSize(); ++i) {
            final double predictedValue = clampedPredictedRow.getValue(i);
            final double targetValue = targetRow.getValue(i);

            final double loss =
                    targetValue * Math.log(predictedValue) + (1.0 - targetValue) * Math.log(1.0 - predictedValue);
            losses.setValue(i, loss);
        }

        return -CustomMath.mean(losses);
    }

    private static DataList clampDataList(final DataList list) {
        final DataList clampedList = new DataList(list.getDataListSize());

        for (int i = 0; i < clampedList.getDataListSize(); ++i) {
            clampedList.setValue(
                    i,
                    Math.clamp(list.getValue(i), BinaryCrossEntropy.CLAMP_VALUE, 1 - BinaryCrossEntropy.CLAMP_VALUE)
            );
        }

        return clampedList;
    }

    @Override
    public String toString() {
        return "Binary cross entropy";
    }
}
