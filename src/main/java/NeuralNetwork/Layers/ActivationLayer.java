package NeuralNetwork.Layers;

import NeuralNetwork.ActivationFunctions.IActivationFunction;
import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import Utilities.CustomMath;

public class ActivationLayer extends AbstractLayer {
    private final IActivationFunction activationFunction;

    public ActivationLayer(final IActivationFunction activationFunction) {
        super();
        this.activationFunction = activationFunction;
    }

    @Override
    protected DataList forward(final DataList inputRow) {
        return this.activationFunction.apply(inputRow);
    }

    @Override
    public Batch backward() {
        final Batch ignore = new Batch();
        return this.calculateGradientWithRespectToInputs(ignore);
    }

    @Override
    public Batch backward(final Batch softmaxOutput) {
        for (int i = 0; i < softmaxOutput.getRowsSize(); ++i) {
            final DataList softmaxOutputRow = softmaxOutput.getRow(i);
            final int size = softmaxOutputRow.getDataListSize();

            final Batch batch = new Batch();

            for (int rowIndex = 0; rowIndex < size; ++rowIndex) {
                final DataList row = new DataList(size);

                for (int columnIndex = 0; columnIndex < size; ++columnIndex) {
                        row.setValue(
                                columnIndex,
                                softmaxOutputRow.getValue(rowIndex) * softmaxOutputRow.getValue(columnIndex)
                        );
                }

                batch.addRow(row);
            }

            final Batch diagonalMatrix = ActivationLayer.getDiagonalMatrix(softmaxOutputRow);
            final Batch jacobianMatrix = CustomMath.subtractBatches(diagonalMatrix, batch);
        }

        return null;
    }

    @Override
    protected Batch calculateGradientWithRespectToBiases(Batch inputGradientBatch) {
        throw new UnsupportedOperationException("Activation layer cannot calculate gradient with respect to biases");
    }

    @Override
    protected Batch calculateGradientWithRespectToWeights(Batch inputGradientBatch) {
        throw new UnsupportedOperationException("Activation layer cannot calculate gradient with respect to weights");
    }

    @Override
    protected Batch calculateGradientWithRespectToInputs(final Batch ignore) {
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

    public static Batch getDiagonalMatrix(final DataList list) {
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
