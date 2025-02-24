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

    public Double[] calculateOutputs(final Double[] inputs) {
        final Double[] outputs = new Double[this.neurons.size()];

        for (int neuronIndex = 0; neuronIndex < this.neurons.size(); ++neuronIndex) {
            outputs[neuronIndex] = this.neurons.get(neuronIndex).calculateOutput(inputs);
        }

        return outputs;
    }
}
