package NeuralNetwork.Layers;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.LossFunctions.CategoricalCrossEntropy;
import NeuralNetwork.LossFunctions.ILossFunction;
import NeuralNetwork.BuildingBlocks.GradientStruct;

public class LossLayer extends LayerBase {
    private final ILossFunction lossFunction;

    public LossLayer(final ILossFunction lossFunction) {
        super();
        this.lossFunction = lossFunction;
    }

    @Override
    public void forward(final Batch inputBatch) {
        this.setSavedInputBatch(inputBatch);

        final Batch predictedBatch = inputBatch;
        final Batch savedTargetBatch = this.getSavedTargetBatch();

        final DataList outputRow = new DataList(predictedBatch.getRowsSize());

        for (int i = 0; i < predictedBatch.getRowsSize(); ++i) {
            final DataList predictedRow = predictedBatch.getRow(i);
            final DataList targetRow = savedTargetBatch.getRow(i);

            final double loss = this.lossFunction.loss(predictedRow, targetRow);
            outputRow.setValue(i, loss);
        }

        final Batch outputBatch = new Batch();
        outputBatch.addRow(outputRow);
        this.setSavedOutputBatch(outputBatch);
    }

    @Override
    public void backward(final GradientStruct inputGradientStruct) {
        if (!inputGradientStruct.gradientStructIsEmpty()) {
            throw new IllegalArgumentException("Loss layer input gradient struct must be empty");
        }
        this.setSavedInputGradientStruct(inputGradientStruct);

        final GradientStruct outputGradientStruct = new GradientStruct();
        outputGradientStruct.setGradientWithRespectToInputs(
                this.resolveCustomCalculateGradientWithRespectToInputs()
        );
        this.setSavedOutputGradientStruct(outputGradientStruct);
    }

    private Batch resolveCustomCalculateGradientWithRespectToInputs() {
        if (this.lossFunction instanceof final CategoricalCrossEntropy categoricalCrossEntropy) {
            return categoricalCrossEntropy.backward(
                    this.getSavedOutputBatch(),
                    this.getSavedTargetBatch()
            );
        } else {
            throw new IllegalArgumentException("Custom backward step is not available for this type of loss function");
        }
    }
}
