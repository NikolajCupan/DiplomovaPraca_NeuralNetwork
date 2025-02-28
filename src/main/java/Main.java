import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.ActivationFunctions.RectifiedLinearUnit;
import NeuralNetwork.ActivationFunctions.Softmax;
import NeuralNetwork.Layers.Common.ActivationLayer;
import NeuralNetwork.Layers.Common.LossLayer;
import NeuralNetwork.Layers.Specific.SoftmaxCategoricalCrossEntropyLayer;
import NeuralNetwork.LossFunctions.CategoricalCrossEntropy;
import NeuralNetwork.BuildingBlocks.Neuron;
import NeuralNetwork.BuildingBlocks.GradientStruct;
import NeuralNetwork.Layers.Common.HiddenLayer;

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

    public static HiddenLayer getLayer1() {
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

    public static HiddenLayer getLayer2() {
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

    public static void main(String[] args) {
        {
            final Batch input = Main.getInput();
            final Batch targetOutput = Main.getOutput();

            final HiddenLayer hiddenLayer1 = Main.getLayer1();
            final ActivationLayer relu = new ActivationLayer(new RectifiedLinearUnit());
            final HiddenLayer hiddenLayer2 = Main.getLayer2();
            final ActivationLayer softmax = new ActivationLayer(new Softmax());
            final LossLayer ccEntropy = new LossLayer(new CategoricalCrossEntropy());

            hiddenLayer1.forward(input);
            relu.forward(hiddenLayer1.getSavedOutputBatch());
            hiddenLayer2.forward(relu.getSavedOutputBatch());
            softmax.forward(hiddenLayer2.getSavedOutputBatch());
            ccEntropy.setSavedTargetBatch(targetOutput);
            ccEntropy.forward(softmax.getSavedOutputBatch());

            ccEntropy.backward(new GradientStruct());
            softmax.backward(ccEntropy.getSavedOutputGradientStruct());
            hiddenLayer2.backward(softmax.getSavedOutputGradientStruct());
            relu.backward(hiddenLayer2.getSavedOutputGradientStruct());
            hiddenLayer1.backward(relu.getSavedOutputGradientStruct());

            System.out.println(hiddenLayer1.getSavedOutputGradientStruct());
        }

        System.out.println("\n\n\n");

        {
            final Batch input = Main.getInput();
            final Batch targetOutput = Main.getOutput();

            final HiddenLayer hiddenLayer1 = Main.getLayer1();
            final ActivationLayer relu = new ActivationLayer(new RectifiedLinearUnit());
            final HiddenLayer hiddenLayer2 = Main.getLayer2();
            final SoftmaxCategoricalCrossEntropyLayer softmaxCCE = new SoftmaxCategoricalCrossEntropyLayer();

            hiddenLayer1.forward(input);
            relu.forward(hiddenLayer1.getSavedOutputBatch());
            hiddenLayer2.forward(relu.getSavedOutputBatch());
            softmaxCCE.setSavedTargetBatch(targetOutput);
            softmaxCCE.forward(hiddenLayer2.getSavedOutputBatch());

            softmaxCCE.backward(new GradientStruct());
            hiddenLayer2.backward(softmaxCCE.getSavedOutputGradientStruct());
            relu.backward(hiddenLayer2.getSavedOutputGradientStruct());
            hiddenLayer1.backward(relu.getSavedOutputGradientStruct());

            System.out.println(hiddenLayer1.getSavedOutputGradientStruct());
        }
    }
}
