package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.DataRow;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Softmax implements IActivationFunction {
    @Override
    public DataRow apply(final DataRow inputRow) {
        final Double[] expValues = new Double[inputRow.getDataRowSize()];
        final double max = Softmax.getMax(inputRow);

        for (int i = 0; i < inputRow.getDataRowSize(); ++i) {
            final double value = inputRow.getValue(i);
            expValues[i] = Math.exp(value - max);
        }

        final DataRow outputRow = new DataRow(inputRow.getDataRowSize());
        final double sum = Softmax.getSum(expValues);

        for (int i = 0; i < inputRow.getDataRowSize(); ++i) {
            outputRow.setValue(
                    i,
                    expValues[i] / sum
            );
        }

        return outputRow;
    }

    private static double getMax(final DataRow inputRow) {
        final Double[] values = inputRow.getDataRowValues();
        final List<Double> list = Arrays.asList(values);
        return Collections.max(list);
    }

    private static double getSum(final Double[] inputRow) {
        double sum = 0.0;
        for (final Double value : inputRow) {
            sum += value;
        }

        return sum;
    }
}
