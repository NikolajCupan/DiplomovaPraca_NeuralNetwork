import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.NeuralNetwork;
import Utilities.Factory;

public class Main {
    private static void run() {
        final Batch trainInput = new Batch();
        final Batch trainTarget = new Batch();

        final Batch testInput = new Batch();
        final Batch testTarget = new Batch();

        Factory.fillBatches(trainInput, trainTarget, testInput, testTarget);


        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork();
        neuralNetwork.train(trainInput, trainTarget, 5, 128, 1, 100);
        neuralNetwork.test(testInput, testTarget);
    }

    private static void run2() {
        final Batch trainInput = Factory.getTrainInputBatch();
        final Batch trainTarget = Factory.getTrainTargetBatch();

        final Batch testInput = Factory.getTestInputBatch();
        final Batch testTarget = Factory.getTestTargetBatch();


        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork2();
        neuralNetwork.train(trainInput, trainTarget, 300, Integer.MAX_VALUE, 50, Integer.MAX_VALUE);
        neuralNetwork.test(testInput, testTarget);
        neuralNetwork.predict(testInput);
    }

    private static void run3() {
        final Batch trainInput = Factory.getTrainInputBatchForClassification();
        final Batch trainTarget = Factory.getTrainTargetBatchForClassification();

        final Batch testInput = Factory.getTestInputBatchForClassification();
        final Batch testTarget = Factory.getTestTargetBatchForClassification();


        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork3();
        neuralNetwork.train(trainInput, trainTarget, 1000, Integer.MAX_VALUE, 100, Integer.MAX_VALUE);


        neuralNetwork.test(testInput, testTarget);
        final Batch predictionsFromTest = neuralNetwork.getPredictedBatch();
        final Batch predictionsFromPredict = neuralNetwork.predict(testInput);
    }

    public static void main(String[] args) {
        final long startTime = System.currentTimeMillis();

        // Main.run();
        // Main.run2();
        Main.run3();

        final long endTime = System.currentTimeMillis();
        System.out.println("Time needed: " + (endTime - startTime) + " ms");
    }
}
