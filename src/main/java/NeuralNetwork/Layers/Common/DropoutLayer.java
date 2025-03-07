package NeuralNetwork.Layers.Common;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.BuildingBlocks.GradientStruct;
import NeuralNetwork.Layers.LayerBase;

import java.util.Optional;
import java.util.Random;

public class DropoutLayer extends LayerBase {
    private Optional<Batch> keepMaskBatch;

    private final double keepRate;
    private final Random random;

    public DropoutLayer(final double keepRate) {
        super();

        this.keepMaskBatch = Optional.empty();
        this.keepRate = keepRate;
        this.random = new Random();
    }

    public DropoutLayer(final double keepRate, final long seed) {
        super();

        this.keepMaskBatch = Optional.empty();
        this.keepRate = keepRate;
        this.random = new Random(seed);
    }

    public double getKeepRate() {
        return this.keepRate;
    }

    public Batch getKeepMaskBatch() {
        if (this.keepMaskBatch.isEmpty()) {
            throw new IllegalArgumentException("Dropout layer saved keep mask batch is empty");
        }

        return this.keepMaskBatch.get();
    }

    public void setKeepMaskBatch(final Batch keepMaskBatch) {
        if (this.keepMaskBatch.isPresent()) {
            throw new IllegalArgumentException("Dropout layer already has keep mask batch set");
        }

        this.keepMaskBatch = Optional.of(keepMaskBatch);
    }

    @Override
    public void clearState() {
        super.clearState();
        this.keepMaskBatch = Optional.empty();
    }

    @Override
    public void forward(final Batch inputBatch) {
        this.setSavedInputBatch(inputBatch);

        final Batch outputBatch = new Batch();
        final Batch keepMask = new Batch();

        final int inputRowSize = inputBatch.getRow(0).getDataListSize();

        for (int rowIndex = 0; rowIndex < inputBatch.getRowsSize(); ++rowIndex) {
            final DataList inputRow = inputBatch.getRow(rowIndex);
            final DataList outputRow = new DataList(inputRowSize);
            final DataList keepMaskRow = new DataList(inputRowSize);

            for (int i = 0; i < inputRowSize; ++i) {
                final double rand = this.random.nextDouble();
                double transformedRand;

                if (rand < this.keepRate) {
                    // Keep the value
                    transformedRand = 1.0 / this.keepRate;
                } else {
                    // Drop the value
                    transformedRand = 0.0;
                }

                final double originalValue = inputRow.getValue(i);
                final double calculatedValue = originalValue * transformedRand;

                outputRow.setValue(i, calculatedValue);
                keepMaskRow.setValue(i, transformedRand);
            }

            outputBatch.addRow(outputRow);
            keepMask.addRow(keepMaskRow);
        }

        this.setKeepMaskBatch(keepMask);
        this.setSavedOutputBatch(outputBatch);
    }

    @Override
    public void backward(final GradientStruct inputGradientStruct) {
        this.setSavedInputGradientStruct(inputGradientStruct);

        final Batch keepMask = this.getKeepMaskBatch();
        final Batch calculatedGradient = new Batch();

        final Batch gradientWRTInputs = inputGradientStruct.getGradientWithRespectToInputs();

        for (int rowIndex = 0; rowIndex < gradientWRTInputs.getRowsSize(); ++rowIndex) {
            final DataList gradientRow = gradientWRTInputs.getRow(rowIndex);
            final DataList keepMaskRow = keepMask.getRow(rowIndex);

            final DataList outputGradientRow = new DataList(gradientRow.getDataListSize());

            for (int i = 0; i < gradientRow.getDataListSize(); ++i) {
                outputGradientRow.setValue(
                        i,
                        gradientRow.getValue(i) * keepMaskRow.getValue(i)
                );
            }

            calculatedGradient.addRow(outputGradientRow);
        }

        final GradientStruct gradientStruct = new GradientStruct();
        gradientStruct.setGradientWithRespectToInputs(calculatedGradient);
        this.setSavedOutputGradientStruct(gradientStruct);
    }

    @Override
    public boolean isCompatible(final LayerBase previousLayer) {
        return previousLayer instanceof ActivationLayer;
    }
}
