package NeuralNetwork.Layers.Special;

import NeuralNetwork.ActivationFunctions.Softmax;
import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.BuildingBlocks.GradientStruct;
import NeuralNetwork.Layers.Common.ActivationLayer;
import NeuralNetwork.Layers.Common.HiddenLayer;
import NeuralNetwork.Layers.Common.LossLayer;
import NeuralNetwork.Layers.IAccuracyLayerBase;
import NeuralNetwork.Layers.ILossLayerBase;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.LossFunctions.CategoricalCrossEntropy;
import Utilities.CustomMath;

public class SoftmaxCategoricalCrossEntropyLayer extends LayerBase implements IAccuracyLayerBase, ILossLayerBase {
    private final ActivationLayer softmaxActivationLayer;
    private final LossLayer categoricalCrossEntropyLossLayer;

    public SoftmaxCategoricalCrossEntropyLayer() {
        super();

        this.softmaxActivationLayer = new ActivationLayer(new Softmax());
        this.categoricalCrossEntropyLossLayer = new LossLayer(new CategoricalCrossEntropy());
    }

    @Override
    public double getAccuracy() {
        return this.categoricalCrossEntropyLossLayer.getAccuracy();
    }

    @Override
    public double getLoss() {
        return this.categoricalCrossEntropyLossLayer.getLoss();
    }

    @Override
    public void clearState() {
        super.clearState();

        this.softmaxActivationLayer.clearState();
        this.categoricalCrossEntropyLossLayer.clearState();
    }

    @Override
    public void forward(final Batch inputBatch) {
        this.setSavedInputBatch(inputBatch);

        this.softmaxActivationLayer.forward(inputBatch);

        this.categoricalCrossEntropyLossLayer.setSavedTargetBatch(this.getSavedTargetBatch());
        this.categoricalCrossEntropyLossLayer.forward(this.softmaxActivationLayer.getSavedOutputBatch());

        this.setSavedOutputBatch(this.categoricalCrossEntropyLossLayer.getSavedOutputBatch());
    }

    @Override
    public void backward(final GradientStruct inputGradientStruct) {
        if (!inputGradientStruct.gradientStructIsEmpty()) {
            throw new IllegalArgumentException("Input gradient struct in softmax categorical cross entropy layer must be empty");
        }
        this.setSavedInputGradientStruct(inputGradientStruct);

        final Batch savedOutputBatchSoftmax = this.softmaxActivationLayer.getSavedOutputBatch();
        final Batch savedTargetBatchCCE = this.categoricalCrossEntropyLossLayer.getSavedTargetBatch();

        final int savedOutputBatchSoftmaxRowsSize = savedOutputBatchSoftmax.getRowsSize();
        final int savedOutputBatchSoftmaxColumnsSize = savedOutputBatchSoftmax.getColumnsSize();

        final Batch gradientWRTInputs = new Batch();

        for (int rowIndex = 0; rowIndex < savedOutputBatchSoftmaxRowsSize; ++rowIndex) {
            final DataList savedOutputBatchSoftmaxRow = savedOutputBatchSoftmax.getRow(rowIndex);
            final DataList savedTargetBatchCCERow = savedTargetBatchCCE.getRow(rowIndex);

            final int targetRowArgMax = CustomMath.argMax(savedTargetBatchCCERow);

            final DataList gradientWRTInputsRow = new DataList(savedOutputBatchSoftmaxColumnsSize);
            for (int columnIndex = 0; columnIndex < savedOutputBatchSoftmaxColumnsSize; ++columnIndex) {
                double calculatedValue = savedOutputBatchSoftmaxRow.getValue(columnIndex);

                if (columnIndex == targetRowArgMax) {
                    calculatedValue -= 1;
                }

                gradientWRTInputsRow.setValue(columnIndex, calculatedValue / savedOutputBatchSoftmaxRowsSize);
            }

            gradientWRTInputs.addRow(gradientWRTInputsRow);
        }

        final GradientStruct gradientStruct = new GradientStruct();
        gradientStruct.setGradientWithRespectToInputs(gradientWRTInputs);
        this.setSavedOutputGradientStruct(gradientStruct);
    }

    @Override
    public boolean isCompatible(final LayerBase previousLayer) {
        return previousLayer instanceof HiddenLayer;
    }

    @Override
    public String toString() {
        return "{\n\tActivation loss layer" +
                "\n" + this.softmaxActivationLayer +
                "\n" + this.categoricalCrossEntropyLossLayer +
                "\n}";
    }
}
