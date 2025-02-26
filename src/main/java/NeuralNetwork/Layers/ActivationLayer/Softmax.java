package NeuralNetwork.Layers.ActivationLayer;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import Utilities.CustomMath;
import Utilities.GradientStruct;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Softmax implements IActivationFunction {
    @Override
    public GradientStruct backward(final Batch inputGradientBatch, final Batch IGNORED_savedInputBatch, final Batch savedOutputBatch) {
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

            final Batch diagonalMatrix = Softmax.getDiagonalMatrix(softmaxOutputRow);
            final Batch jacobianMatrix = CustomMath.subtractBatches(diagonalMatrix, batch);

            final DataList inputGradientRow = inputGradientBatch.getRow(i);
            final DataList resultRow = new DataList(jacobianMatrix.getRowsSize());

            for (int jacobianMatrixRowIndex = 0; jacobianMatrixRowIndex < jacobianMatrix.getRowsSize(); ++jacobianMatrixRowIndex) {
                resultRow.setValue(jacobianMatrixRowIndex,
                        CustomMath.dotProduct(jacobianMatrix.getRow(jacobianMatrixRowIndex).getDataListRawValues(), inputGradientRow.getDataListRawValues())
                );
            }

            gradientWRTInputs.addRow(resultRow);
        }

        final GradientStruct gradientStruct = new GradientStruct();
        gradientStruct.setGradientWithRespectToInputs(gradientWRTInputs);
        return gradientStruct;
    }

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
