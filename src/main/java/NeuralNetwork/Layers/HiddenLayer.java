package NeuralNetwork.Layers;

import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.BuildingBlocks.Neuron;
import Utilities.CustomMath;
import NeuralNetwork.BuildingBlocks.GradientStruct;

import java.util.ArrayList;
import java.util.List;

public class HiddenLayer extends LayerBase {
    private final List<Neuron> neurons;

    public HiddenLayer() {
        super();
        this.neurons = new ArrayList<>();
    }

    public void addNeuron(final Neuron neuron) {
        if (!this.neurons.isEmpty()) {
            final int weightsSize = this.neurons.getFirst().getWeightsSize();

            if (neuron.getWeightsSize() != weightsSize) {
                throw new IllegalArgumentException("New neuron weights size [" + neuron.getWeightsSize() + "] is not equal to current weights size [" + weightsSize + "]");
            }
        }

        this.neurons.add(neuron);
    }

    @Override
    public void forward(final Batch inputBatch) {
        this.setSavedInputBatch(inputBatch);

        final Batch outputBatch = new Batch();

        for (int rowIndex = 0; rowIndex < inputBatch.getRowsSize(); ++rowIndex) {
            final DataList inputRow = inputBatch.getRow(rowIndex);

            final DataList calculatedRow = this.forwardInputRow(inputRow);
            outputBatch.addRow(calculatedRow);
        }

        this.setSavedOutputBatch(outputBatch);
    }

    private DataList forwardInputRow(final DataList inputRow) {
        final DataList outputRow = new DataList(this.neurons.size());

        for (int neuronIndex = 0; neuronIndex < this.neurons.size(); ++neuronIndex) {
            final double neuronOutput = this.neurons.get(neuronIndex).forward(inputRow);
            outputRow.setValue(neuronIndex, neuronOutput);
        }

        return outputRow;
    }

    @Override
    public void backward(final GradientStruct inputGradientStruct) {
        this.setSavedInputGradientStruct(inputGradientStruct);

        final Batch gradientWRTBiases = this.calculateGradientWithRespectToBiases(inputGradientStruct);
        final Batch gradientWRTWeights = this.calculateGradientWithRespectToWeights(inputGradientStruct);
        final Batch gradientWRTInputs = this.calculateGradientWithRespectToInputs(inputGradientStruct);

        final GradientStruct outputGradientBatch = new GradientStruct();
        outputGradientBatch.setGradientWithRespectToBiases(gradientWRTBiases);
        outputGradientBatch.setGradientWithRespectToWeights(gradientWRTWeights);
        outputGradientBatch.setGradientWithRespectToInputs(gradientWRTInputs);
        this.setSavedOutputGradientStruct(outputGradientBatch);
    }

    private Batch calculateGradientWithRespectToBiases(final GradientStruct inputGradientStruct) {
        final Batch inputGradientWRTInputs = inputGradientStruct.getGradientWithRespectToInputs();

        final int inputGradientColumnsSize = inputGradientWRTInputs.getColumnsSize();
        final DataList calculatedGradient = new DataList(inputGradientColumnsSize);

        for (int columnIndex = 0; columnIndex < inputGradientColumnsSize; ++columnIndex) {
            final DataList gradientColumn = inputGradientWRTInputs.getColumn(columnIndex);
            calculatedGradient.setValue(
                    columnIndex,
                    CustomMath.sum(gradientColumn)
            );
        }

        final Batch gradientBatch = new Batch();
        gradientBatch.addRow(calculatedGradient);
        return gradientBatch;
    }

    private Batch calculateGradientWithRespectToWeights(final GradientStruct inputGradientStruct) {
        final Batch outputGradientBatch = new Batch();

        final Batch savedInputBatch = this.getSavedInputBatch();
        final int inputBatchColumnsSize = savedInputBatch.getColumnsSize();

        final Batch inputGradientWRTInputs = inputGradientStruct.getGradientWithRespectToInputs();
        final int inputGradientColumnsSize = inputGradientWRTInputs.getColumnsSize();

        for (int inputColumnIndex = 0; inputColumnIndex < inputBatchColumnsSize; ++inputColumnIndex) {
            final DataList inputBatchColumn = savedInputBatch.getColumn(inputColumnIndex);

            final DataList resultRow = new DataList(inputGradientColumnsSize);

            for (int inputGradientIndex = 0; inputGradientIndex < inputGradientColumnsSize; ++inputGradientIndex) {
                final DataList inputGradientColumn = inputGradientWRTInputs.getColumn(inputGradientIndex);

                final double result = CustomMath.dotProduct(inputGradientColumn.getDataListRawValues(), inputBatchColumn.getDataListRawValues());
                resultRow.setValue(
                        inputGradientIndex,
                        result
                );
            }

            outputGradientBatch.addRow(resultRow);
        }

        return outputGradientBatch;
    }

    private Batch calculateGradientWithRespectToInputs(final GradientStruct inputGradientStruct) {
        final Batch outputGradientBatch = new Batch();
        final Batch inputGradientWRTInputs = inputGradientStruct.getGradientWithRespectToInputs();

        for (int rowIndex = 0; rowIndex < inputGradientWRTInputs.getRowsSize(); ++rowIndex) {
            final DataList inputGradientRow = inputGradientWRTInputs.getRow(rowIndex);
            outputGradientBatch.addRow(this.calculateGradientWithRespectToInputs(inputGradientRow));
        }

        return outputGradientBatch;
    }

    private DataList calculateGradientWithRespectToInputs(final DataList inputGradient) {
        final int neuronsSize = this.neurons.size();
        if (neuronsSize != inputGradient.getDataListSize()) {
            throw new IllegalArgumentException("Input gradient size [" + inputGradient.getDataListSize() + "] is not equal to neurons size [" + neuronsSize + "]");
        }

        // Number of connections between one neuron in current layer and all inputs from previous layer
        final int weightsSize = this.neurons.getFirst().getWeightsSize();

        final DataList outputGradient = new DataList(weightsSize);

        for (int columnIndex = 0; columnIndex < weightsSize; ++columnIndex) {
            final DataList columnWeights = this.getColumnWeights(columnIndex);
            outputGradient.setValue(
                    columnIndex,
                    CustomMath.dotProduct(inputGradient.getDataListRawValues(), columnWeights.getDataListRawValues())
            );
        }

        return outputGradient;
    }

    private DataList getColumnWeights(final int columnIndex) {
        if (this.neurons.isEmpty()) {
            throw new IllegalArgumentException("Neuron list is empty");
        }

        // Number of connections between one input from previous layer and all neurons in current layer
        final int connectionsSize = this.neurons.size();

        final DataList columnWeights = new DataList(connectionsSize);

        for (int rowIndex = 0; rowIndex < connectionsSize; ++rowIndex) {
            final Neuron neuron = this.neurons.get(rowIndex);
            final double weight = neuron.getWeight(columnIndex);

            columnWeights.setValue(rowIndex, weight);
        }

        return columnWeights;
    }

    @Override
    public String toString() {
        if (this.neurons.isEmpty()) {
            return "{ Layer: empty }";
        }

        final StringBuilder builder = new StringBuilder();

        builder.append("{\n\tHidden layer: input [");
        builder.append(this.neurons.getLast().getWeightsSize());
        builder.append("], output [");
        builder.append(this.neurons.size());
        builder.append("]");

        for (final Neuron neuron : this.neurons) {
            builder.append("\n\t\t").append(neuron);
        }

        builder.append("\n}");

        return builder.toString();
    }
}
