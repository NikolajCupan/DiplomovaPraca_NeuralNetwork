package Utilities;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;

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

    public static int argMax(final DataList list) {
        if (list.isEmpty()) {
            throw new IllegalArgumentException("Data list is empty");
        }

        final Double[] dataListValues = list.getDataListRawValues();

        double max = dataListValues[0];
        int maxIndex = 0;

        for (int i = 0; i < dataListValues.length; ++i) {
            if (dataListValues[i] > max) {
                max = dataListValues[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public static DataList mean(final Batch batch) {
        final DataList outputList = new DataList(batch.getRowsSize());

        for (int rowIndex = 0; rowIndex < batch.getRowsSize(); ++rowIndex) {
            final DataList row = batch.getRow(rowIndex);
            final double rowMean = CustomMath.mean(row);

            outputList.setValue(rowIndex, rowMean);
        }

        return outputList;
    }

    public static double mean(final DataList list) {
        if (list.isEmpty()) {
            throw new IllegalArgumentException("Data list is empty");
        }

        final double sum = CustomMath.sum(list);
        final double size = list.getDataListSize();

        return sum / size;
    }

    public static double sum(final DataList list) {
        return CustomMath.sum(list.getDataListRawValues());
    }

    public static double sum(final Double[] values) {
        double sum = 0.0;

        for (final Double value : values) {
            sum += value;
        }

        return sum;
    }
}
