import NeuralNetwork.ActivationFunctions.RectifiedLinearUnit;
import NeuralNetwork.ActivationFunctions.Softmax;
import NeuralNetwork.Batch;
import NeuralNetwork.DataRow;
import NeuralNetwork.Layers.ActivationLayer;
import NeuralNetwork.Layers.Layer;
import NeuralNetwork.Neuron;
import Utilities.SeedGenerator;

public class Main {
    private static final SeedGenerator SEED_GENERATOR = new SeedGenerator(420);

    public static Layer getLayer1() {
        final Neuron neuron1 = new Neuron(
                4, Main.SEED_GENERATOR.getSeed()
        );

        final Neuron neuron2 = new Neuron(
                4, Main.SEED_GENERATOR.getSeed()
        );

        final Neuron neuron3 = new Neuron(
                4, Main.SEED_GENERATOR.getSeed()
        );

        final Layer layer = new Layer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        return layer;
    }

    public static ActivationLayer getActivationLayer1() {
        return new ActivationLayer(
                new RectifiedLinearUnit()
        );
    }

    public static Layer getLayer2() {
        final Neuron neuron1 = new Neuron(
                3, Main.SEED_GENERATOR.getSeed()
        );

        final Neuron neuron2 = new Neuron(
                3, Main.SEED_GENERATOR.getSeed()
        );

        final Neuron neuron3 = new Neuron(
                3, Main.SEED_GENERATOR.getSeed()
        );

        final Layer layer = new Layer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        return layer;
    }

    public static ActivationLayer getActivationLayer2() {
        return new ActivationLayer(
                new Softmax()
        );
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
        final Batch inputBatch = getBatch();

        final Layer layer1 = getLayer1();
        final ActivationLayer activationLayer1 = getActivationLayer1();
        final Layer layer2 = getLayer2();
        final ActivationLayer activationLayer2 = getActivationLayer2();

        final Batch outputBatch1 = layer1.calculateOutputBatch(inputBatch);
        final Batch outputBatch2 = activationLayer1.calculateOutputBatch(outputBatch1);
        final Batch outputBatch3 = layer2.calculateOutputBatch(outputBatch2);
        final Batch outputBatch4 = activationLayer2.calculateOutputBatch(outputBatch3);

        System.out.println(inputBatch);
        System.out.println(outputBatch1);
        System.out.println(outputBatch2);
        System.out.println(outputBatch3);
        System.out.println(outputBatch4);
    }
}
