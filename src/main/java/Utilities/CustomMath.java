package Utilities;

import NeuralNetwork.DataRow;

public class CustomMath {
    public static Double dotProduct(
            final Double[] left,
            final Double[] right
    ) {
        if (left.length != right.length) {
            throw new IllegalArgumentException("Size of left input [" + left.length + "] is not equal to size of right input [" + right.length + "]");
        }

        Double result = 0.0;
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

        Double max = dataRowValues[0];
        int maxIndex = 0;

        for (int i = 0; i < dataRowValues.length; ++i) {
            if (dataRowValues[i] > max) {
                max = dataRowValues[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
