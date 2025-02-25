package NeuralNetwork.Layers;

import NeuralNetwork.DataRow;
import NeuralNetwork.Neuron;
import Utilities.CustomMath;

import java.util.ArrayList;
import java.util.List;

public class Layer implements ILayer {
    private final List<Neuron> neurons;

    public Layer() {
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

    public Double[] calculateGradient(final Double[] gradient) {
        final int neuronsSize = this.neurons.size();
        if (neuronsSize != gradient.length) {
            throw new IllegalArgumentException("Gradient size [" + gradient.length + "] is not equal to neurons size [" + neuronsSize + "]");
        }

        // Number of connections between one neuron in current layer and all inputs from previous layer
        final int weightsSize = this.neurons.getFirst().getWeightsSize();

        final Double[] newGradient = new Double[weightsSize];

        for (int inputIndex = 0; inputIndex < weightsSize; ++inputIndex) {
            final Double[] inputConnectionWeights = this.getWeights(inputIndex);
            newGradient[inputIndex] = CustomMath.dotProduct(gradient, inputConnectionWeights);
        }

        return newGradient;
    }

    private Double[] getWeights(final int inputIndex) {
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
        final Double[] weights = new Double[connectionsSize];

        for (int neuronIndex = 0; neuronIndex < connectionsSize; ++neuronIndex) {
            final Neuron neuron = this.neurons.get(neuronIndex);
            final double weight = neuron.getWeight(inputIndex);

            weights[neuronIndex] = weight;
        }

        return weights;
    }

    @Override
    public DataRow calculateOutputRow(final DataRow inputRow) {
        final DataRow outputRow = new DataRow(this.neurons.size());

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
