package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.DataRow;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Softmax implements IActivationFunction {
    @Override
    public DataRow apply(final DataRow inputRow) {
        final Double[] expValues = new Double[inputRow.getDataRowSize()];
        final Double max = Softmax.getMax(inputRow);

        for (int i = 0; i < inputRow.getDataRowSize(); ++i) {
            final Double value = inputRow.getValue(i);
            expValues[i] = Math.exp(value - max);
        }

        final DataRow outputRow = new DataRow(inputRow.getDataRowSize());
        final Double sum = Softmax.getSum(expValues);

        for (int i = 0; i < inputRow.getDataRowSize(); ++i) {
            outputRow.setValue(
                    i,
                    expValues[i] / sum
            );
        }

        return outputRow;
    }

    private static Double getMax(final DataRow inputRow) {
        final Double[] values = inputRow.getDataRowValues();
        final List<Double> list = Arrays.asList(values);
        return Collections.max(list);
    }

    private static Double getSum(final Double[] inputRow) {
        Double sum = 0.0;
        for (final Double value : inputRow) {
            sum += value;
        }

        return sum;
    }
}
