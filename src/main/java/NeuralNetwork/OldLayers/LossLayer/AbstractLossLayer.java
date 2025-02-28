//package NeuralNetwork.OldLayers.LossLayer;
//
//import NeuralNetwork.BuildingBlocks.Batch;
//import NeuralNetwork.BuildingBlocks.DataList;
//
//import java.util.Optional;
//
//public abstract class AbstractLossLayer {
//    private Optional<Batch> savedPredictedBatch;
//    private Optional<Batch> savedTargetBatch;
//
//    protected AbstractLossLayer() {
//        this.savedPredictedBatch = Optional.empty();
//        this.savedTargetBatch = Optional.empty();
//    }
//
//    public Batch getSavedPredictedBatch() {
//        if (this.savedPredictedBatch.isEmpty()) {
//            throw new IllegalArgumentException("Layer has not processed an input batch yet");
//        }
//
//        return this.savedPredictedBatch.get();
//    }
//
//    public Batch getSavedTargetBatch() {
//        if (this.savedTargetBatch.isEmpty()) {
//            throw new IllegalArgumentException("Layer has not processed an input batch yet");
//        }
//
//        return this.savedTargetBatch.get();
//    }
//
//    public Batch forward(final Batch predictedBatch, final Batch targetBatch) {
//        if (this.savedPredictedBatch.isPresent() || this.savedTargetBatch.isPresent()) {
//            throw new IllegalArgumentException("Layer already processed an input batch");
//        }
//        this.savedPredictedBatch = Optional.of(predictedBatch);
//        this.savedTargetBatch = Optional.of(targetBatch);
//
//        final DataList outputRow = new DataList(predictedBatch.getRowsSize());
//
//        for (int i = 0; i < predictedBatch.getRowsSize(); ++i) {
//            final DataList predictedRow = predictedBatch.getRow(i);
//            final DataList targetRow = targetBatch.getRow(i);
//
//            final double loss = this.forward(predictedRow, targetRow);
//            outputRow.setValue(i, loss);
//        }
//
//        final Batch batch = new Batch();
//        batch.addRow(outputRow);
//        return batch;
//    }
//
//    private DataList forward(final Batch predictedBatch, final Integer[] targetIndexes) {
//        final DataList outputRow = new DataList(predictedBatch.getRowsSize());
//
//        for (int i = 0; i < predictedBatch.getRowsSize(); ++i) {
//            final DataList predictedRow = predictedBatch.getRow(i);
//            final int targetIndex = targetIndexes[i];
//
//            final double loss = this.forward(predictedRow, targetIndex);
//            outputRow.setValue(i, loss);
//        }
//
//        return outputRow;
//    }
//
//    public abstract Batch backward();
//
//    protected abstract double forward(final DataList predictedRow, final DataList targetRow);
//    protected abstract double forward(final DataList predictedRow, final int targetIndex);
//}
