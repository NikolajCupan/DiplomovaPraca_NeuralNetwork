package Utilities;

import NeuralNetwork.Batch;
import NeuralNetwork.DataRow;

public class CustomMath {
    public static double dotProduct(
            final Double[] left,
            final Double[] right
    ) {
        if (left.length != right.length) {
            throw new IllegalArgumentException("Size of left input [" + left.length + "] is not equal to size of right input [" + right.length + "]");
        }

        double result = 0.0;
        for (int i = 0; i < left.length; ++i) {
            result += left[i] * right[i];
        }

        return result;
    }

    public static int argMax(final DataRow dataRow) {
        if (dataRow.isEmpty()) {
            throw new IllegalArgumentException("Data row is empty");
        }

        final Double[] dataRowValues = dataRow.getDataRowValues();

        double max = dataRowValues[0];
        int maxIndex = 0;

        for (int i = 0; i < dataRowValues.length; ++i) {
            if (dataRowValues[i] > max) {
                max = dataRowValues[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public static DataRow mean(final Batch batch) {
        final DataRow outputRow = new DataRow(batch.getBatchSize());

        for (int i = 0; i < batch.getBatchSize(); ++i) {
            final DataRow inputRow = batch.getInputRow(i);
            final double inputRowMean = CustomMath.mean(inputRow);

            outputRow.setValue(i, inputRowMean);
        }

        return outputRow;
    }

    public static double mean(final DataRow dataRow) {
        if (dataRow.isEmpty()) {
            throw new IllegalArgumentException("Data row is empty");
        }

        final double sum = CustomMath.sum(dataRow);
        final double size = dataRow.getDataRowSize();

        return sum / size;
    }

    public static double sum(final DataRow dataRow) {
        double sum = 0.0;

        final Double[] dataRowValues = dataRow.getDataRowValues();
        for (final Double dataRowValue : dataRowValues) {
            sum += dataRowValue;
        }

        return sum;
    }
}
