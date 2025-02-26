package NeuralNetwork.Layers.ActivationLayer;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import Utilities.GradientStruct;

public class RectifiedLinearUnit implements IActivationFunction {
    @Override
    public GradientStruct backward(final Batch IGNORED_inputGradientBatch, final Batch savedInputBatch, final Batch IGNORED_savedOutputBatch) {
        final Batch gradientWRTInputs = new Batch();

        for (int rowIndex = 0; rowIndex < savedInputBatch.getRowsSize(); ++rowIndex) {
            final DataList inputBatchRow = savedInputBatch.getRow(rowIndex);
            gradientWRTInputs.addRow(this.calculateGradientWithRespectToInputs(inputBatchRow));
        }

        final GradientStruct gradientStruct = new GradientStruct();
        gradientStruct.setGradientWithRespectToInputs(gradientWRTInputs);
        return gradientStruct;
    }

    private DataList calculateGradientWithRespectToInputs(final DataList inputGradient) {
        final DataList outputList = new DataList(inputGradient.getDataListSize());

        for (int columnIndex = 0; columnIndex < inputGradient.getDataListSize(); ++columnIndex) {
            final double derivative = this.calculateDerivative(
                    inputGradient.getValue(columnIndex)
            );

            outputList.setValue(columnIndex, derivative);
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

    @Override
    public double calculateDerivative(final double value) {
        return Math.max(0, value);
    }
}
