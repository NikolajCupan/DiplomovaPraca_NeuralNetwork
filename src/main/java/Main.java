public class Main {
    public static Layer getLayer1() {
        final Neuron neuron1 = new Neuron(
                2.0, new Double[]{ 0.2, 0.8, -0.5, 1.0 }
        );

        final Neuron neuron2 = new Neuron(
                3.0, new Double[]{ 0.5, -0.91, 0.26, -0.5 }
        );

        final Neuron neuron3 = new Neuron(
                0.5, new Double[]{ -0.26, -0.27, 0.17, 0.87 }
        );

        final Layer layer = new Layer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        return layer;
    }

    public static Layer getLayer2() {
        final Neuron neuron1 = new Neuron(
                -1.0, new Double[]{ 0.1, -0.14, 0.5 }
        );

        final Neuron neuron2 = new Neuron(
                2.0, new Double[]{ -0.5, 0.12, -0.33 }
        );

        final Neuron neuron3 = new Neuron(
                -0.5, new Double[]{ -0.44, 0.73, -0.13 }
        );

        final Layer layer = new Layer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        return layer;
    }

    public static void main(String[] args) {
        final Layer layer1 = getLayer1();
        final Layer layer2 = getLayer2();

        final Double[] outputs1 = layer1.calculateOutputs(new Double[]{ 1.0, 2.0, 3.0, 2.5 });
        Helper.printArray(outputs1);

        final Double[] outputs2 = layer1.calculateOutputs(new Double[]{ 2.0, 5.0, -1.0, 2.0 });
        Helper.printArray(outputs2);

        final Double[] outputs3 = layer1.calculateOutputs(new Double[]{ -1.5, 2.7, 3.3, -0.8 });
        Helper.printArray(outputs3);
    }
}
