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
    public Batch calculateGradientWithRespectToBiases(Batch inputGradientBatch) {
        throw new UnsupportedOperationException("Activation layer cannot calculate gradient with respect to biases");
    }

    @Override
    public Batch calculateGradientWithRespectToWeights(Batch inputGradientBatch) {
        throw new UnsupportedOperationException("Activation layer cannot calculate gradient with respect to weights");
    }

    @Override
    public Batch calculateGradientWithRespectToInputs(final Batch ignore) {
        final Batch outputGradientBatch = new Batch();

        for (int rowIndex = 0; rowIndex < this.getSavedInputBatch().getRowsSize(); ++rowIndex) {
            final DataList inputBatchRow = this.getSavedInputBatch().getRow(rowIndex);
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
    protected DataList forward(final DataList inputRow) {
        return this.activationFunction.apply(inputRow);
    }
}
