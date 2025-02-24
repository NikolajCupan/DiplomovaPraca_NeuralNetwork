public class Main {
    public static void main(String[] args) {
        final Neuron neuron1 = new Neuron(
            2, new double[]{ 0.2, 0.8, -0.5, 1.0 }
        );

        final Neuron neuron2 = new Neuron(
            3, new double[]{ 0.5, -0.91, 0.26, -0.5 }
        );

        final Neuron neuron3 = new Neuron(
            0.5, new double[]{ -0.26, -0.27, 0.17, 0.87 }
        );

        final Layer layer = new Layer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        final double[] outputs = layer.calculateOutputs(new double[]{ 1.0, 2.0, 3.0, 2.5 });
        for (int i = 0; i < outputs.length; ++i) {
            System.out.println(outputs[i]);
        }
    }
}
