import GUI.GUI;
import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.NeuralNetwork;
import Utilities.Factory;

import java.util.Arrays;

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
        neuralNetwork.train(trainInput, trainTarget, 250, Integer.MAX_VALUE, 100, Integer.MAX_VALUE);


        neuralNetwork.test(testInput, testTarget);
        final Batch predictionsFromTest = neuralNetwork.getPredictedBatch();
        final Batch predictionsFromPredict = neuralNetwork.predict(testInput);
    }

    private static void run4() {
        final Batch trainInput = Factory.getTrainInputBatchForRegression();
        final Batch trainTarget = Factory.getTrainTargetBatchForRegression();

        final Batch testInput = Factory.getTestInputBatchForRegression();
        final Batch testTarget = Factory.getTestTargetBatchForRegression();


        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork4();
        neuralNetwork.train(trainInput, trainTarget, 25000, Integer.MAX_VALUE, 250, Integer.MAX_VALUE);

        final Batch predictedTrain = neuralNetwork.predict(trainInput);
        final Batch predictedTest = neuralNetwork.predict(testInput);


        final DataList forecast = neuralNetwork.forecast(testInput.getRow(testInput.getRowsSize() - 1), 100);



        final GUI gui = new GUI();
        gui.show(Arrays.asList(
                new GUI.ChartData(trainTarget.getColumn(0), "train_target", "blue", 0),
                new GUI.ChartData(testTarget.getColumn(0), "test_target", "cyan", trainTarget.getRowsSize() - 1),
                new GUI.ChartData(predictedTrain.getColumn(0), "predicted_train", "orange", 0),
                new GUI.ChartData(predictedTest.getColumn(0), "predicted_test", "red", trainTarget.getRowsSize() - 1),
                new GUI.ChartData(forecast, "forecast", "green", testTarget.getRowsSize() + trainTarget.getRowsSize() - 1)
        ));
    }

    public static void main(String[] args) {
        final long startTime = System.currentTimeMillis();

        // Main.run();
        // Main.run2();
        // Main.run3();
        Main.run4();

        final long endTime = System.currentTimeMillis();
        System.out.println("Time needed: " + (endTime - startTime) + " ms");
    }
}
