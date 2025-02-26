package NeuralNetwork.Layers.ActivationLayer;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import Utilities.GradientStruct;

public class RectifiedLinearUnit implements IActivationFunction {
    @Override
    public GradientStruct backward(final Batch inputGradientBatch, final Batch savedInputBatch, final Batch IGNORED_savedOutputBatch) {
        final Batch gradientWRTInputs = new Batch();

        for (int rowIndex = 0; rowIndex < savedInputBatch.getRowsSize(); ++rowIndex) {
            final DataList inputGradientRow = inputGradientBatch.getRow(rowIndex);
            final DataList savedInputRow = savedInputBatch.getRow(rowIndex);
            gradientWRTInputs.addRow(this.calculateGradientWithRespectToInputs(inputGradientRow, savedInputRow));
        }

        final GradientStruct gradientStruct = new GradientStruct();
        gradientStruct.setGradientWithRespectToInputs(gradientWRTInputs);
        return gradientStruct;
    }

    private DataList calculateGradientWithRespectToInputs(final DataList inputGradientRow, final DataList savedInputRow) {
        final DataList outputList = new DataList(inputGradientRow.getDataListSize());

        for (int columnIndex = 0; columnIndex < inputGradientRow.getDataListSize(); ++columnIndex) {
            final double inputGradientValue = inputGradientRow.getValue(columnIndex);
            final double reluInputValue = savedInputRow.getValue(columnIndex);

            if (reluInputValue > 0) {
                outputList.setValue(columnIndex, inputGradientValue);
            } else {
                outputList.setValue(columnIndex, 0);
            }
        }

        return outputList;
    }

    @Override
    public DataList apply(final DataList inputList) {
        final DataList outputList = new DataList(inputList.getDataListSize());

        for (int i = 0; i < inputList.getDataListSize(); ++i) {
            outputList.setValue(
                    i,
                    Math.max(0, inputList.getValue(i))
            );
        }

        return outputList;
    }
}
