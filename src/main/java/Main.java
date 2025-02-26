import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import NeuralNetwork.Layers.ActivationLayer.ActivationLayer;
import NeuralNetwork.Layers.ActivationLayer.RectifiedLinearUnit;
import NeuralNetwork.Layers.ActivationLayer.Softmax;
import NeuralNetwork.Layers.LossLayer.AbstractLossLayer;
import NeuralNetwork.Layers.LossLayer.CategoricalCrossEntropy;
import NeuralNetwork.Layers.NormalLayer.Layer;
import NeuralNetwork.Neuron;
import Utilities.GradientStruct;

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

        final DataList output1 = new DataList(new Double[]{ 1.0, 0.0, 0.0, 0.0, 0.0 });
        final DataList output2 = new DataList(new Double[]{ 1.0, 0.0, 0.0, 0.0, 0.0 });
        final DataList output3 = new DataList(new Double[]{ 0.0, 1.0, 0.0, 0.0, 0.0 });

        batch.addRow(output1);
        batch.addRow(output2);
        batch.addRow(output3);

        return batch;
    }

    public static Layer getLayer1() {
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

    public static Layer getLayer2() {
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

        final Layer layer = new Layer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);
        layer.addNeuron(neuron4);
        layer.addNeuron(neuron5);

        return layer;
    }

    public static void main(String[] args) {
        final Batch input = Main.getInput();
        final Batch targetOutput = Main.getOutput();

        final Layer layer1 = Main.getLayer1();
        final ActivationLayer relu = new ActivationLayer(new RectifiedLinearUnit());
        final Layer layer2 = Main.getLayer2();
        final ActivationLayer softmax = new ActivationLayer(new Softmax());
        final AbstractLossLayer ccEntropy = new CategoricalCrossEntropy();

        final Batch layer1Output = layer1.forward(input);
        final Batch reluOutput = relu.forward(layer1Output);
        final Batch layer2Output = layer2.forward(reluOutput);
        final Batch softmaxOutput = softmax.forward(layer2Output);
        final Batch ccEntropyOutput = ccEntropy.forward(softmaxOutput, targetOutput);

        System.out.println(input);
        System.out.println(targetOutput + "\n\n\n");
        System.out.println(layer1Output);
        System.out.println(reluOutput);
        System.out.println(layer2Output);
        System.out.println(softmaxOutput);
        System.out.println(ccEntropyOutput + "\n\n\n");

        final Batch ccEntropyGradient = ccEntropy.backward();
        final GradientStruct softmaxGradient = softmax.backward(ccEntropyGradient);
        final GradientStruct layer2Gradient = layer2.backward(softmaxGradient.getGradientWithRespectToInputs());
        final GradientStruct reluGradient = relu.backward(layer2Gradient.getGradientWithRespectToInputs());
        final GradientStruct layer1Gradient = layer1.backward(reluGradient.getGradientWithRespectToInputs());

        System.out.println(softmaxGradient.getGradientWithRespectToInputs());
    }
}
