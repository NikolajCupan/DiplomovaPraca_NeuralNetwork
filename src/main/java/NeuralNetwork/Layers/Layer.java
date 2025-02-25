package NeuralNetwork.Layers;

import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import NeuralNetwork.Neuron;
import Utilities.CustomMath;

import java.util.ArrayList;
import java.util.List;

public class Layer extends AbstractLayer {
    private final List<Neuron> neurons;

    public Layer() {
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

    public Batch calculateGradientWithRespectToWeights(final Batch inputGradientBatch) {
        final Batch outputGradientBatch = new Batch();

        final Batch inputBatch = this.getInputBatch();
        final int inputBatchColumnsSize = inputBatch.getColumnsSize();

        final int inputGradientColumnsSize = inputGradientBatch.getColumnsSize();

        for (int inputColumnIndex = 0; inputColumnIndex < inputBatchColumnsSize; ++inputColumnIndex) {
            final DataList inputBatchColumn = inputBatch.getColumn(inputColumnIndex);

            final DataList resultRow = new DataList(inputGradientColumnsSize);

            for (int inputGradientIndex = 0; inputGradientIndex < inputGradientColumnsSize; ++inputGradientIndex) {
                final DataList inputGradientColumn = inputGradientBatch.getColumn(inputGradientIndex);

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

    public Batch calculateGradientWithRespectToInputs(final Batch inputGradientBatch) {
        final Batch outputGradientBatch = new Batch();

        for (int i = 0; i < inputGradientBatch.getRowsSize(); ++i) {
            final DataList inputGradientRow = inputGradientBatch.getRow(i);
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
    protected DataList calculateOutputRow(final DataList inputRow) {
        final DataList outputRow = new DataList(this.neurons.size());

        for (int neuronIndex = 0; neuronIndex < this.neurons.size(); ++neuronIndex) {
            final double neuronOutput = this.neurons.get(neuronIndex).calculateOutput(inputRow);
            outputRow.setValue(neuronIndex, neuronOutput);
        }

        return outputRow;
    }

    @Override
    public String toString() {
        if (this.neurons.isEmpty()) {
            return "{ Layer: empty }";
        }

        final StringBuilder builder = new StringBuilder();

        builder.append("{ Layer: input [");
        builder.append(this.neurons.getLast().getWeightsSize());
        builder.append("], output [");
        builder.append(this.neurons.size());
        builder.append("] }");

        return builder.toString();
    }
}
