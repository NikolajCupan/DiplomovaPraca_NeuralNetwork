package NeuralNetwork.Layers.Common;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.ActivationFunctions.IActivationFunction;
import NeuralNetwork.ActivationFunctions.Softmax;
import NeuralNetwork.BuildingBlocks.GradientStruct;
import NeuralNetwork.Layers.LayerBase;

public class ActivationLayer extends LayerBase {
    private final IActivationFunction activationFunction;

    public ActivationLayer(final IActivationFunction activationFunction) {
        super();
        this.activationFunction = activationFunction;
    }

    @Override
    public void forward(final Batch inputBatch) {
        this.setSavedInputBatch(inputBatch);

        final Batch outputBatch = new Batch();

        for (int rowIndex = 0; rowIndex < inputBatch.getRowsSize(); ++rowIndex) {
            final DataList inputRow = inputBatch.getRow(rowIndex);
            final DataList calculatedRow = this.activationFunction.activate(inputRow);

            outputBatch.addRow(calculatedRow);
        }

        this.setSavedOutputBatch(outputBatch);
    }

    @Override
    public void backward(final GradientStruct inputGradientStruct) {
        this.setSavedInputGradientStruct(inputGradientStruct);
        final GradientStruct outputGradientStruct = new GradientStruct();

        if (this.activationFunction.requiresCustomBackwardStep()) {
            outputGradientStruct.setGradientWithRespectToInputs(
                    this.resolveCustomCalculateGradientWithRespectToInputs(inputGradientStruct)
            );
        } else {
            final Batch savedInputBatch = this.getSavedInputBatch();
            final Batch inputGradientWRTInputs = inputGradientStruct.getGradientWithRespectToInputs();

            final Batch gradientWRTInputs = new Batch();

            for (int rowIndex = 0; rowIndex < savedInputBatch.getRowsSize(); ++rowIndex) {
                final DataList inputGradientRow = inputGradientWRTInputs.getRow(rowIndex);
                final DataList savedInputRow = savedInputBatch.getRow(rowIndex);
                gradientWRTInputs.addRow(this.calculateGradientWithRespectToInputs(inputGradientRow, savedInputRow));
            }

            outputGradientStruct.setGradientWithRespectToInputs(gradientWRTInputs);
        }

        this.setSavedOutputGradientStruct(outputGradientStruct);
    }

    private DataList calculateGradientWithRespectToInputs(final DataList inputGradientRow, final DataList savedInputRow) {
        final DataList outputRow = new DataList(inputGradientRow.getDataListSize());

        for (int columnIndex = 0; columnIndex < inputGradientRow.getDataListSize(); ++columnIndex) {
            final double inputGradientValue = inputGradientRow.getValue(columnIndex);
            final double savedInputValue = savedInputRow.getValue(columnIndex);

            final double derivative = this.activationFunction.derivative(savedInputValue);
            outputRow.setValue(columnIndex, derivative * inputGradientValue);
        }

        return outputRow;
    }

    private Batch resolveCustomCalculateGradientWithRespectToInputs(final GradientStruct inputGradientStruct) {
        if (this.activationFunction instanceof final Softmax softmaxActivationFunction) {
            final Batch savedOutputBatch = this.getSavedOutputBatch();
            return softmaxActivationFunction.backward(
                    inputGradientStruct,
                    savedOutputBatch
            );
        } else {
            throw new IllegalArgumentException("Custom backward step is not available for this type of activation function");
        }
    }

    @Override
    public String toString() {
        return "{\n\tActivation layer: activation function [" +
                this.activationFunction +
                "]\n}";
    }
}
