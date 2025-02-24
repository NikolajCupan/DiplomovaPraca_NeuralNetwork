import NeuralNetwork.Batch;
import NeuralNetwork.DataRow;
import NeuralNetwork.Layer;
import NeuralNetwork.Neuron;

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

    public static Batch getBatch() {
        final Batch batch = new Batch();

        final DataRow inputs1 = new DataRow(new Double[]{ 1.0, 2.0, 3.0, 2.5 });
        final DataRow inputs2 = new DataRow(new Double[]{ 2.0, 5.0, -1.0, 2.0 });
        final DataRow inputs3 = new DataRow(new Double[]{ -1.5, 2.7, 3.3, -0.8 });

        batch.addInputRow(inputs1);
        batch.addInputRow(inputs2);
        batch.addInputRow(inputs3);

        return batch;
    }

    public static void main(String[] args) {
        final Layer layer1 = getLayer1();
        final Layer layer2 = getLayer2();

        final Batch inputBatch = getBatch();
        final Batch outputBatch1 = layer1.calculateOutputBatch(inputBatch);
        final Batch outputBatch2 = layer2.calculateOutputBatch(outputBatch1);

        System.out.println(inputBatch);
        System.out.println(outputBatch1);
        System.out.println(outputBatch2);
    }
}
