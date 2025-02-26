import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import NeuralNetwork.Layers.ActivationLayer.ActivationLayer;
import NeuralNetwork.Layers.ActivationLayer.Softmax;
import NeuralNetwork.Layers.LossLayer.AbstractLossLayer;
import NeuralNetwork.Layers.LossLayer.CategoricalCrossEntropy;
import NeuralNetwork.Layers.NormalLayer.Layer;
import NeuralNetwork.Neuron;

public class Main {
    public static Batch getInput() {
        final Batch batch = new Batch();

        final DataList input1 = new DataList(new Double[]{ 1.0, 2.0, 3.0, 2.5 });
        final DataList input2 = new DataList(new Double[]{ 2.0, 5.0, -1.0, 2.0 });
        final DataList input3 = new DataList(new Double[]{ -1.5, 2.7, 3.3, -0.8 });

        batch.addRow(input1);
        batch.addRow(input2);
        batch.addRow(input3);

        return batch;
    }

    public static Batch getOutput() {
        final Batch batch = new Batch();

        final DataList input1 = new DataList(new Double[]{ 1.0, 0.0, 0.0 });
        final DataList input2 = new DataList(new Double[]{ 1.0, 0.0, 0.0 });
        final DataList input3 = new DataList(new Double[]{ 0.0, 1.0, 0.0 });

        batch.addRow(input1);
        batch.addRow(input2);
        batch.addRow(input3);

        return batch;
    }

    public static Layer getLayer() {
        final Neuron neuron1 = new Neuron(
                2.0, new DataList(new Double[]{ 0.2, 0.8, -0.5, 1.0 })
        );
        final Neuron neuron2 = new Neuron(
                3.0, new DataList(new Double[]{ 0.5, -0.91, 0.26, -0.5 })
        );
        final Neuron neuron3 = new Neuron(
                0.5, new DataList(new Double[]{ -0.26, -0.27, 0.17, 0.87 })
        );

        final Layer layer = new Layer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        return layer;
    }

    public static void main(String[] args) {
        final Layer layer = Main.getLayer();
        final ActivationLayer softmax = new ActivationLayer(new Softmax());
        final AbstractLossLayer ccEntropy = new CategoricalCrossEntropy();

        final Batch input = Main.getInput();
        final Batch targetOutput = Main.getOutput();

        final Batch layerOutput = layer.forward(input);
        final Batch softmaxOutput = softmax.forward(layerOutput);

        final DataList loss = ccEntropy.calculate(softmaxOutput, targetOutput);

        System.out.println(input);
        System.out.println(targetOutput + "\n\n\n");
        System.out.println(layerOutput);
        System.out.println(softmaxOutput + "\n\n\n");
        System.out.println(loss);
    }
}
