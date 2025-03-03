package NeuralNetwork.Layers.Common;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.Layers.IAccuracyLayerBase;
import NeuralNetwork.Layers.ILossLayerBase;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.LossFunctions.BinaryCrossEntropy;
import NeuralNetwork.LossFunctions.CategoricalCrossEntropy;
import NeuralNetwork.LossFunctions.ILossFunction;
import NeuralNetwork.BuildingBlocks.GradientStruct;

public class LossLayer extends LayerBase implements IAccuracyLayerBase, ILossLayerBase  {
    private final ILossFunction lossFunction;

    public LossLayer(final ILossFunction lossFunction) {
        super();
        this.lossFunction = lossFunction;
    }

    @Override
    public double getAccuracy() {
        if (this.lossFunction instanceof final CategoricalCrossEntropy categoricalCrossEntropy) {
            return categoricalCrossEntropy.getAccuracy(
                    this.getSavedInputBatch(),
                    this.getSavedTargetBatch()
            );
        } else if (this.lossFunction instanceof final BinaryCrossEntropy binaryCrossEntropy) {
            return binaryCrossEntropy.getAccuracy();
        } else {
            throw new RuntimeException("Loss layer cannot be used to calculate accuracy");
        }
    }

    @Override
    public double getLoss() {
        if (this.lossFunction instanceof final CategoricalCrossEntropy categoricalCrossEntropy) {
            return categoricalCrossEntropy.getLoss(
                   this.getSavedOutputBatch()
            );
        } else if (this.lossFunction instanceof final BinaryCrossEntropy binaryCrossEntropy) {
            return binaryCrossEntropy.getLoss();
        } else {
            throw new RuntimeException("Loss layer cannot be used to calculate loss");
        }
    }

    @Override
    public void forward(final Batch inputBatch) {
        this.setSavedInputBatch(inputBatch);

        final Batch savedTargetBatch = this.getSavedTargetBatch();

        final DataList outputRow = new DataList(inputBatch.getRowsSize());

        for (int i = 0; i < inputBatch.getRowsSize(); ++i) {
            final DataList predictedRow = inputBatch.getRow(i);
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
            throw new IllegalArgumentException("Input gradient struct in loss layer must be empty");
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
                    this.getSavedInputBatch(),
                    this.getSavedTargetBatch()
            );
        } else if (this.lossFunction instanceof final BinaryCrossEntropy binaryCrossEntropy) {
            return binaryCrossEntropy.backward(
                    this.getSavedInputBatch(),
                    this.getSavedTargetBatch()
            );
        } else {
            throw new IllegalArgumentException("Custom backward step is not available for this type of loss function");
        }
    }

    @Override
    public boolean isCompatible(final LayerBase previousLayer) {
        return previousLayer instanceof ActivationLayer;
    }

    @Override
    public String toString() {
        return "{\n\tLoss layer: loss function [" +
                this.lossFunction +
                "]\n}";
    }
}
