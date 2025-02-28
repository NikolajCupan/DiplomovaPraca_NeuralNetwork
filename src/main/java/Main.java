import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.ActivationFunctions.RectifiedLinearUnit;
import NeuralNetwork.Layers.Common.ActivationLayer;
import NeuralNetwork.Layers.Special.SoftmaxCategoricalCrossEntropyLayer;
import NeuralNetwork.BuildingBlocks.Neuron;
import NeuralNetwork.Layers.Common.HiddenLayer;
import NeuralNetwork.NeuralNetwork;

public class Main {
    public static HiddenLayer getHiddenLayer1() {
        final Neuron neuron1 = new Neuron(
                2.0, new DataList(new Double[]{ 0.2, 0.8, -0.5, 1.0 })
        );
        final Neuron neuron2 = new Neuron(
                3.0, new DataList(new Double[]{ 0.5, -0.91, 0.26, -0.5 })
        );
        final Neuron neuron3 = new Neuron(
                0.5, new DataList(new Double[]{ -0.26, -0.27, 0.17, 0.87 })
        );

        final HiddenLayer layer = new HiddenLayer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        return layer;
    }

    public static HiddenLayer getHiddenLayer2() {
        final Neuron neuron1 = new Neuron(
                1.0, new DataList(new Double[]{ 0.1, -0.3, 0.5 })
        );
        final Neuron neuron2 = new Neuron(
                1.5, new DataList(new Double[]{ -0.2, 0.4, -0.6 })
        );
        final Neuron neuron3 = new Neuron(
                0.8, new DataList(new Double[]{ 0.3, -0.7, 0.1 })
        );
        final Neuron neuron4 = new Neuron(
                1.2, new DataList(new Double[]{ -0.5, 0.6, 0.2 })
        );
        final Neuron neuron5 = new Neuron(
                0.9, new DataList(new Double[]{ 0.4, -0.2, -0.1 })
        );

        final HiddenLayer layer = new HiddenLayer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);
        layer.addNeuron(neuron4);
        layer.addNeuron(neuron5);

        return layer;
    }

    public static NeuralNetwork getNeuralNetwork() {
        final HiddenLayer hiddenLayer1 = Main.getHiddenLayer1();
        final ActivationLayer relu = new ActivationLayer(new RectifiedLinearUnit());
        final HiddenLayer hiddenLayer2 = Main.getHiddenLayer2();
        final SoftmaxCategoricalCrossEntropyLayer softmaxCCE = new SoftmaxCategoricalCrossEntropyLayer();

        final NeuralNetwork neuralNetwork = new NeuralNetwork(4);
        neuralNetwork.addHiddenLayer(hiddenLayer1);
        neuralNetwork.addActivationLayer(relu);
        neuralNetwork.addHiddenLayer(hiddenLayer2);
        neuralNetwork.addSpecialLayer(softmaxCCE);
        return neuralNetwork;
    }

    public static Batch getInputBatch() {
        final Batch batch = new Batch();

        final DataList input1 = new DataList(new Double[]{ 1.0, 2.0, 3.0, 2.5 });
        final DataList input2 = new DataList(new Double[]{ 2.0, 5.0, -1.0, 2.0 });
        final DataList input3 = new DataList(new Double[]{ -1.5, 2.7, 3.3, -0.8 });

        batch.addRow(input1);
        batch.addRow(input2);
        batch.addRow(input3);

        return batch;
    }

    public static Batch getTargetBatch() {
        final Batch batch = new Batch();

        final DataList output1 = new DataList(new Double[]{ 1.0, 0.0, 0.0, 0.0, 0.0 });
        final DataList output2 = new DataList(new Double[]{ 1.0, 0.0, 0.0, 0.0, 0.0 });
        final DataList output3 = new DataList(new Double[]{ 0.0, 1.0, 0.0, 0.0, 0.0 });

        batch.addRow(output1);
        batch.addRow(output2);
        batch.addRow(output3);

        return batch;
    }

    public static void main(String[] args) {
        final Batch inputBatch = Main.getInputBatch();
        final Batch targetBatch = Main.getTargetBatch();

        final NeuralNetwork neuralNetwork = Main.getNeuralNetwork();
        neuralNetwork.forward(inputBatch, targetBatch);
        neuralNetwork.backward();

        int x = 100;
    }
}
