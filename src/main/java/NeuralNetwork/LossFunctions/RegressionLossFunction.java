package NeuralNetwork.LossFunctions;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import Utilities.CustomMath;

public abstract class RegressionLossFunction implements ILossFunction {
    private final double maxPercentageDifference;

    protected RegressionLossFunction(final double maxPercentageDifference) {
        this.maxPercentageDifference = maxPercentageDifference;
    }

    @Override
    public double getAccuracyForPrinting(final Batch predictedBatch, final Batch targetBatch) {
        assert(predictedBatch.getRowsSize() == targetBatch.getRowsSize());

        final double targetBatchStandardDeviation = CustomMath.standardDeviation(targetBatch);
        final double precision = targetBatchStandardDeviation / 250.0;

        final DataList correctPredictionsList = new DataList(predictedBatch.getRowsSize());

        for (int rowIndex = 0; rowIndex < predictedBatch.getRowsSize(); ++rowIndex) {
            final DataList predictedRow = predictedBatch.getRow(rowIndex);
            final DataList targetRow = targetBatch.getRow(rowIndex);

            double correctPredictions = 0.0;

            for (int i = 0; i < predictedRow.getDataListSize(); ++i) {
                final double prediction = predictedRow.getValue(i);
                final double target = targetRow.getValue(i);

                final double percentualDifference = CustomMath.percentualDifference(prediction, target);

                if (percentualDifference <= this.maxPercentageDifference) {
                    correctPredictions += 1.0;
                }
            }

            correctPredictionsList.setValue(rowIndex, correctPredictions / predictedRow.getDataListSize());
        }

        return CustomMath.mean(correctPredictionsList);
    }
}
