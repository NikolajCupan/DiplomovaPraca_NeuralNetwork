import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.NeuralNetwork;
import Utilities.Factory;

public class Main {
    private static void run() {
        final Batch trainInputBatch = Factory.getTrainInputBatch();
        final Batch trainTargetBatch = Factory.getTargetBatch();

        final Batch testInputBatch = Factory.getTestInputBatch();
        final Batch testTargetBatch = Factory.getTestTargetBatch();


        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork();
        neuralNetwork.train(trainInputBatch, trainTargetBatch, 1000, 100);
        neuralNetwork.test(testInputBatch, testTargetBatch);
    }

    public static void main(String[] args) {
        final long startTime = System.currentTimeMillis();

        Main.run();

        final long endTime = System.currentTimeMillis();
        System.out.println("Time needed: " + (endTime - startTime) + " ms");
    }
}
