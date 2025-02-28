package Utilities;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

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

    public static Batch subtractBatches(final Batch left, final Batch right) {
        if (
                left.getRowsSize() != right.getRowsSize()
                || left.getColumnsSize() != right.getColumnsSize()
        ) {
            throw new IllegalArgumentException("Size of batches do not match, left [" + left.getRowsSize() + " x " + left.getColumnsSize()
                    + "], right [" + right.getRowsSize() + " x " + right.getColumnsSize() + "]");
        }

        final Batch outputBatch = new Batch();

        for (int rowIndex = 0; rowIndex < left.getRowsSize(); ++rowIndex) {
            final DataList leftRow = left.getRow(rowIndex);
            final DataList rightRow = right.getRow(rowIndex);

            final DataList calculatedRow = new DataList(left.getColumnsSize());

            for (int columnIndex = 0; columnIndex < left.getColumnsSize(); ++columnIndex) {
                calculatedRow.setValue(
                        columnIndex,
                        leftRow.getValue(columnIndex) - rightRow.getValue(columnIndex)
                );
            }

            outputBatch.addRow(calculatedRow);
        }

        return outputBatch;
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

    public static double max(final DataList inputList) {
        final Double[] values = inputList.getDataListRawValues();
        final List<Double> list = Arrays.asList(values);
        return Collections.max(list);
    }

    // INPUT:         OUTPUT:
    // [ 1, 2, 3 ] -> [ 1, 0, 0 ]
    //                [ 0, 2, 0 ]
    //                [ 0, 0, 3 ]
    public static Batch diagonalMatrix(final DataList list) {
        final Batch matrix = new Batch();
        final int size = list.getDataListSize();

        for (int i = 0; i < size; ++i) {
            final DataList row = new DataList(size);
            row.fill(0);

            row.setValue(i, list.getValue(i));
            matrix.addRow(row);
        }

        return matrix;
    }
}
