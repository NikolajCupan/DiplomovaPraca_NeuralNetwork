import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.NeuralNetworkHelper;
import Utilities.Factory;

public class Main {
    private static void run() {
        final Batch trainInput = Factory.getTrainInputBatch();
        final Batch trainTarget = Factory.getTrainTargetBatch();

        final Batch testInput = Factory.getTestInputBatch();
        final Batch testTarget = Factory.getTestTargetBatch();

        // Factory.fillBatches(trainInput, trainTarget, testInput, testTarget);


        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork();
        neuralNetwork.train(trainInput, trainTarget, 50, Integer.MAX_VALUE, 25, Integer.MAX_VALUE);
        neuralNetwork.test(testInput, testTarget);


        NeuralNetworkHelper.saveToFile(neuralNetwork, "first.txt");
        final NeuralNetwork neuralNetwork2 = NeuralNetworkHelper.loadFromFile("first.txt");

        int x = 100;
    }

    public static void main(String[] args) {
        final long startTime = System.currentTimeMillis();

        Main.run();

        final long endTime = System.currentTimeMillis();
        System.out.println("Time needed: " + (endTime - startTime) + " ms");
    }
}
