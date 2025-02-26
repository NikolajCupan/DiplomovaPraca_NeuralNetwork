package NeuralNetwork.Layers;

import NeuralNetwork.ActivationFunctions.IActivationFunction;
import NeuralNetwork.Batch;
import NeuralNetwork.DataList;

public class ActivationLayer extends AbstractLayer {
    private final IActivationFunction activationFunction;

    public ActivationLayer(final IActivationFunction activationFunction) {
        super();
        this.activationFunction = activationFunction;
    }

    @Override
    public Batch calculateGradientWithRespectToInputs(final Batch ignore) {
        final Batch outputGradientBatch = new Batch();

        for (int rowIndex = 0; rowIndex < this.getInputBatch().getRowsSize(); ++rowIndex) {
            final DataList inputBatchRow = this.getInputBatch().getRow(rowIndex);
            outputGradientBatch.addRow(this.calculateGradientWithRespectToInputs(inputBatchRow));
        }

        return outputGradientBatch;
    }

    @Override
    protected DataList calculateGradientWithRespectToInputs(final DataList inputGradient) {
        final DataList outputList = new DataList(inputGradient.getDataListSize());

        for (int columnIndex = 0; columnIndex < inputGradient.getDataListSize(); ++columnIndex) {
            final double derivative = this.activationFunction.calculateDerivative(
                    inputGradient.getValue(columnIndex)
            );

            outputList.setValue(columnIndex, derivative);
        }

        return outputList;
    }

    @Override
    protected DataList calculateOutputRow(final DataList inputRow) {
        return this.activationFunction.apply(inputRow);
    }
}
