package NeuralNetwork.Layers.Common;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.Layers.IAccuracyForPrintingGetter;
import NeuralNetwork.Layers.ILossForPrintingGetter;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.LossFunctions.*;
import NeuralNetwork.BuildingBlocks.GradientStruct;

public class LossLayer extends LayerBase implements IAccuracyForPrintingGetter, ILossForPrintingGetter {
    private final ILossFunction lossFunction;

    public LossLayer(final ILossFunction lossFunction) {
        super();
        this.lossFunction = lossFunction;
    }

    public ILossFunction getLossFunction() {
        return this.lossFunction;
    }

    @Override
    public double getAccuracyForPrinting() {
        return this.lossFunction.getAccuracyForPrinting(
                this.getSavedInputBatch(),
                this.getSavedTargetBatch()
        );
    }

    @Override
    public double getLossForPrinting() {
        return this.lossFunction.getLossForPrinting(
                this.getSavedOutputBatch()
        );
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
                this.lossFunction.backward(this.getSavedInputBatch(), this.getSavedTargetBatch()
        ));
        this.setSavedOutputGradientStruct(outputGradientStruct);
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
