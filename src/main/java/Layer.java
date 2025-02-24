import java.util.ArrayList;
import java.util.List;

public class Layer {
    private final List<Neuron> neurons;

    public Layer() {
        this.neurons = new ArrayList<>();
    }

    public void addNeuron(final Neuron neuron) {
        this.neurons.add(neuron);
    }

    public double[] calculateOutputs(final double[] inputs) {
        final double[] outputs = new double[this.neurons.size()];

        for (int neuronIndex = 0; neuronIndex < this.neurons.size(); ++neuronIndex) {
            outputs[neuronIndex] = this.neurons.get(neuronIndex).calculateOutput(inputs);
        }

        return outputs;
    }
}
