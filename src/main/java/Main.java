import NeuralNetwork.ActivationFunctions.Softmax;
import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import NeuralNetwork.Layers.ActivationLayer;
import NeuralNetwork.Layers.Layer;
import NeuralNetwork.LossFunctions.CategoricalCrossEntropy;
import NeuralNetwork.Neuron;

public class Main {
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

    public static ActivationLayer getActivationLayer() {
        return new ActivationLayer(
                new Softmax()
        );
    }

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

    public static Batch getGradient() {
        final Batch batch = new Batch();

        final DataList gradientRow1 = new DataList(new Double[]{ 1.0, 1.0, 1.0 });
        final DataList gradientRow2 = new DataList(new Double[]{ 2.0, 2.0, 2.0 });
        final DataList gradientRow3 = new DataList(new Double[]{ 3.0, 3.0, 3.0 });

        batch.addRow(gradientRow1);
        batch.addRow(gradientRow2);
        batch.addRow(gradientRow3);

        return batch;
    }

    public static CategoricalCrossEntropy getLossLayer() {
        return new CategoricalCrossEntropy();
    }

    public static void main(String[] args) {
        final Batch input = Main.getInput();
        final Batch targetOutput = Main.getOutput();

        final Layer layer = Main.getLayer();
        final ActivationLayer softmax = new ActivationLayer(new Softmax());
        final CategoricalCrossEntropy ccentropy = new CategoricalCrossEntropy();

        final Batch layerOutput = layer.forward(input);
        final Batch softmaxOutput = softmax.forward(layerOutput);
        final DataList ccnetropyOutput = ccentropy.calculate(softmaxOutput, targetOutput);

        final Batch ccentropyGradient = ccentropy.backward(softmaxOutput, targetOutput);

        final Batch smOutput = new Batch();
        smOutput.addRow(new DataList(new Double[]{ 0.7, 0.1, 0.2 }));
        // smOutput.addRow(new DataList(new Double[]{ 0.3, 0.4, 0.3 }));

        final Batch ceOutput = new Batch();
        ceOutput.addRow(new DataList(new Double[]{ 0.5, -0.2, 0.1 }));
        //ceOutput.addRow(new DataList(new Double[]{ -0.1, 0.4, -0.3 }));
        softmax.back(smOutput, ceOutput);


        System.out.println("input");
        System.out.println(input);

        System.out.println("\ntarget output");
        System.out.println(targetOutput);

        System.out.println("\n");

        System.out.println("\nlayer output");
        System.out.println(layerOutput);

        System.out.println("\nsoftmax output");
        System.out.println(softmaxOutput);
        System.out.println("\ncc entropy output");
        System.out.println(ccnetropyOutput);
    }
}
