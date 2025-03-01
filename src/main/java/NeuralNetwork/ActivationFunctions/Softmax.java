package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import Utilities.CustomMath;
import NeuralNetwork.BuildingBlocks.GradientStruct;

public class Softmax implements IActivationFunction {
    @Override
    public boolean requiresCustomBackwardStep() {
        return true;
    }

    @Override
    public DataList activate(final DataList inputList) {
        final double[] expValues = new double[inputList.getDataListSize()];
        final double max = CustomMath.max(inputList);

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

    @Override
    public double derivative(final double input) {
        throw new UnsupportedOperationException("Softmax activation function does not support direct derivative");
    }

    public Batch backward(final GradientStruct inputGradientStruct, final Batch savedOutputBatch) {
        final Batch inputGradientWRTInputs = inputGradientStruct.getGradientWithRespectToInputs();
        final Batch gradientWRTInputs = new Batch();

        for (int i = 0; i < savedOutputBatch.getRowsSize(); ++i) {
            final DataList softmaxOutputRow = savedOutputBatch.getRow(i);
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

            final Batch diagonalMatrix = CustomMath.diagonalMatrix(softmaxOutputRow);
            final Batch jacobianMatrix = CustomMath.subtractBatches(diagonalMatrix, batch);

            final DataList inputGradientRow = inputGradientWRTInputs.getRow(i);
            final DataList resultRow = new DataList(jacobianMatrix.getRowsSize());

            for (int jacobianMatrixRowIndex = 0; jacobianMatrixRowIndex < jacobianMatrix.getRowsSize(); ++jacobianMatrixRowIndex) {
                resultRow.setValue(jacobianMatrixRowIndex,
                        CustomMath.dotProduct(jacobianMatrix.getRow(jacobianMatrixRowIndex).getDataListRawValues(), inputGradientRow.getDataListRawValues())
                );
            }

            gradientWRTInputs.addRow(resultRow);
        }

        return gradientWRTInputs;
    }

    @Override
    public String toString() {
        return "Softmax";
    }
}
