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
            final int weightsSize = this.neurons.getLast().getWeightsSize();

            if (neuron.getWeightsSize() != weightsSize) {
                throw new IllegalArgumentException("New neuron weights size [" + neuron.getWeightsSize() + "] is not equal to current weights size [" + weightsSize + "]");
            }
        }

        this.neurons.add(neuron);
    }

    public Batch calculateGradientWithRespectToWeights(final Batch inputGradientBatch) {
        final Batch outputGradientBatch = new Batch();

        for (int i = 0; i < inputGradientBatch.getColumnsSize(); ++i) {
            final DataList inputGradientColumn = inputGradientBatch.getColumn(i);
            outputGradientBatch.addRow(this.calculateGradientWithRespectToWeights(inputGradientColumn));
        }

        return outputGradientBatch;
    }

    private DataList calculateGradientWithRespectToWeights(final DataList inputGradient) {
        final int neuronsSize = this.neurons.size();
        if (neuronsSize != inputGradient.getDataListSize()) {
            throw new IllegalArgumentException("Input gradient size [" + inputGradient.getDataListSize() + "] is not equal to neurons size [" + neuronsSize + "]");
        }

        // Number of connections between one neuron in current layer and all inputs from previous layer
        final int inputsRowSize = this.getInputsRowSize();

        final DataList outputGradient = new DataList(inputsRowSize);

        for (int inputIndex = 0; inputIndex < inputsRowSize; ++inputIndex) {
            final DataList inputs = this.getInputs(inputIndex);
            outputGradient.setValue(
                    inputIndex,
                    CustomMath.dotProduct(inputs.getDataListRawValues(), inputGradient.getDataListRawValues())
            );
        }

        return outputGradient;
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

        for (int weightIndex = 0; weightIndex < weightsSize; ++weightIndex) {
            final DataList inputConnectionWeights = this.getWeights(weightIndex);
            outputGradient.setValue(
                    weightIndex,
                    CustomMath.dotProduct(inputGradient.getDataListRawValues(), inputConnectionWeights.getDataListRawValues())
            );
        }

        return outputGradient;
    }

    private DataList getWeights(final int inputIndex) {
        if (this.neurons.isEmpty()) {
            throw new IllegalArgumentException("Neuron list is empty");
        }

        // Number of connections between one input from previous layer and all neurons in current layer
        final int connectionsSize = this.neurons.size();

        // Number of connections between one neuron in current layer and all inputs from previous layer
        final int weightsSize = this.neurons.getFirst().getWeightsSize();
        if (inputIndex >= weightsSize) {
            throw new IllegalArgumentException("Input index [" + inputIndex + "] is invalid, weights size [" + weightsSize + "]");
        }

        // Weights associated with the input
        final DataList weights = new DataList(connectionsSize);

        for (int neuronIndex = 0; neuronIndex < connectionsSize; ++neuronIndex) {
            final Neuron neuron = this.neurons.get(neuronIndex);
            final double weight = neuron.getWeight(inputIndex);

            weights.setValue(neuronIndex, weight);
        }

        return weights;
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
