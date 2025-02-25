package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.DataList;
import Utilities.CustomMath;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Softmax implements IActivationFunction {
    @Override
    public DataList apply(final DataList inputList) {
        final Double[] expValues = new Double[inputList.getDataListSize()];
        final double max = Softmax.getMax(inputList);

        for (int i = 0; i < inputList.getDataListSize(); ++i) {
            final double value = inputList.getValue(i);
            expValues[i] = Math.exp(value - max);
        }

        final DataList outputList = new DataList(inputList.getDataListSize());
        final double sum = CustomMath.sum(expValues);

        for (int i = 0; i < inputList.getDataListSize(); ++i) {
            outputList.setValue(
                    i,
                    expValues[i] / sum
            );
        }

        return outputList;
    }

    private static double getMax(final DataList inputList) {
        final Double[] values = inputList.getDataListRawValues();
        final List<Double> list = Arrays.asList(values);
        return Collections.max(list);
    }
}
