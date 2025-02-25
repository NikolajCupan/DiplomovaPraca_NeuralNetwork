import NeuralNetwork.DataRow;
import NeuralNetwork.Layers.Layer;
import NeuralNetwork.Neuron;

public class Main {
    public static Layer getLayer() {
        final Neuron neuron1 = new Neuron(
                1.0, new DataRow(new Double[]{ 0.2, 0.8, -0.5, 1.0 })
        );
        final Neuron neuron2 = new Neuron(
                1.0, new DataRow(new Double[]{ 0.5, -0.91, 0.26, -0.5 })
        );
        final Neuron neuron3 = new Neuron(
                1.0, new DataRow(new Double[]{ -0.26, -0.27, 0.17, 0.87 })
        );

        final Layer layer = new Layer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        return layer;
    }

    public static void main(String[] args) {
        final DataRow inputGradient = new DataRow(new Double[]{ 1.0, 1.0, 1.0 });
        final Layer layer = getLayer();

        final DataRow outputGradient = layer.calculateGradient(inputGradient);
        System.out.println(outputGradient);
    }
}
