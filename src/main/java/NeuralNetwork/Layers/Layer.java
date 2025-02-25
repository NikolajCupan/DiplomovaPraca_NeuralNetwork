package NeuralNetwork.Layers;

import NeuralNetwork.DataRow;
import NeuralNetwork.Neuron;

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
