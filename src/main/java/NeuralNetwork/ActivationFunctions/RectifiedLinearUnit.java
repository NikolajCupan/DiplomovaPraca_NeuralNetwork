package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.DataRow;

public class RectifiedLinearUnit implements IActivationFunction {
    @Override
    public DataRow apply(final DataRow inputRow) {
        final DataRow outputRow = new DataRow(inputRow.getDataRowSize());

        for (int i = 0; i < inputRow.getDataRowSize(); ++i) {
            outputRow.setValue(
                    i,
                    Math.max(0, inputRow.getValue(i))
            );
        }

        return outputRow;
    }
}
