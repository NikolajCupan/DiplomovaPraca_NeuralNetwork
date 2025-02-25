import NeuralNetwork.Batch;
import NeuralNetwork.DataRow;
import NeuralNetwork.Layers.Layer;
import NeuralNetwork.Neuron;

public class Main {
    public static Layer getLayer() {
        final Neuron neuron1 = new Neuron(
                1.0, new DataRow(new Double[]{ 0.2, 0.8, -0.5, 1.0 })
        );
        final Neuron neuron2 = new Neuron(
                1.0, new DataRow(new Double[]{ 0.5, -0.91, 0.26, -0.5 })
        );
        final Neuron neuron3 = new Neuron(
                1.0, new DataRow(new Double[]{ -0.26, -0.27, 0.17, 0.87 })
        );

        final Layer layer = new Layer();
        layer.addNeuron(neuron1);
        layer.addNeuron(neuron2);
        layer.addNeuron(neuron3);

        return layer;
    }

    public static Batch getInputGradientBatch() {
        final Batch batch = new Batch();

        final DataRow gradientRow1 = new DataRow(new Double[]{ 1.0, 1.0, 1.0 });
        final DataRow gradientRow2 = new DataRow(new Double[]{ 2.0, 2.0, 2.0 });
        final DataRow gradientRow3 = new DataRow(new Double[]{ 3.0, 3.0, 3.0 });

        batch.addDataRow(gradientRow1);
        batch.addDataRow(gradientRow2);
        batch.addDataRow(gradientRow3);

        return batch;
    }

    public static void main(String[] args) {
        final Layer layer = Main.getLayer();
        final Batch inputGradientBatch = Main.getInputGradientBatch();

        final Batch outputGradientBatch = layer.calculateGradient(inputGradientBatch);
        System.out.println(outputGradientBatch);
    }
}
