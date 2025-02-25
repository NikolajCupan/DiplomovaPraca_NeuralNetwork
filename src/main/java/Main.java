import NeuralNetwork.Batch;
import NeuralNetwork.DataList;
import NeuralNetwork.Layers.Layer;
import NeuralNetwork.Neuron;

public class Main {
    public static Layer getLayer() {
        final Neuron neuron1 = new Neuron(
                1.0, new DataList(new Double[]{ 0.2, 0.8, -0.5, 1.0 })
        );
        final Neuron neuron2 = new Neuron(
                1.0, new DataList(new Double[]{ 0.5, -0.91, 0.26, -0.5 })
        );
        final Neuron neuron3 = new Neuron(
                1.0, new DataList(new Double[]{ -0.26, -0.27, 0.17, 0.87 })
        );

        final Layer layer = new Layer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        return layer;
    }

    public static Batch getInputDataBatch() {
        final Batch batch = new Batch();

        final DataList gradientRow1 = new DataList(new Double[]{ 1.0, 2.0, 3.0, 2.5 });
        final DataList gradientRow2 = new DataList(new Double[]{ 2.0, 5.0, -1.0, 2.0 });
        final DataList gradientRow3 = new DataList(new Double[]{ -1.5, 2.7, 3.3, -0.8 });

        batch.addRow(gradientRow1);
        batch.addRow(gradientRow2);
        batch.addRow(gradientRow3);

        return batch;
    }

    public static Batch getInputGradientBatch() {
        final Batch batch = new Batch();

        final DataList gradientRow1 = new DataList(new Double[]{ 1.0, 1.0, 1.0 });
        final DataList gradientRow2 = new DataList(new Double[]{ 2.0, 5.0, 8.0 });
        final DataList gradientRow3 = new DataList(new Double[]{ 3.0, 3.0, 3.0 });

        batch.addRow(gradientRow1);
        batch.addRow(gradientRow2);
        batch.addRow(gradientRow3);

        return batch;
    }

    public static void main(String[] args) {
        final Layer layer = Main.getLayer();
        final Batch inputDataBatch = Main.getInputDataBatch();
        final Batch inputGradientBatch = Main.getInputGradientBatch();

        layer.calculateOutputBatch(inputDataBatch);
        final Batch outputGradientBatch = layer.calculateGradientWithRespectToWeights(inputGradientBatch);
        System.out.println(outputGradientBatch);
    }
}
