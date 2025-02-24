package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    private final List<Neuron> neurons;

    public Layer() {
        this.neurons = new ArrayList<>();
    }

    public void addNeuron(final Neuron neuron) {
        if (!this.neurons.isEmpty()) {
            final long weightsSize = this.neurons.getLast().getWeightsSize();

            if (neuron.getWeightsSize() != weightsSize) {
                throw new IllegalArgumentException("New neuron weights size [" + neuron.getWeightsSize() + "] is not equal to current weights size [" + weightsSize + "]");
            }
        }

        this.neurons.add(neuron);
    }

    public Batch calculateOutputBatch(final Batch batch) {
        final Batch outputBatch = new Batch();

        for (int i = 0; i < batch.getBatchSize(); ++i) {
            final DataRow inputRow = batch.getInputRow(i);
            final DataRow calculatedRow = this.calculateOutputRow(inputRow);

            outputBatch.addInputRow(calculatedRow);
        }

        return outputBatch;
    }

    public DataRow calculateOutputRow(final DataRow inputRow) {
        final DataRow outputRow = new DataRow(this.neurons.size());

        for (int neuronIndex = 0; neuronIndex < this.neurons.size(); ++neuronIndex) {
            final Double neuronOutput = this.neurons.get(neuronIndex).calculateOutput(inputRow);
            outputRow.setValue(neuronIndex, neuronOutput);
        }

        return outputRow;
    }
}
