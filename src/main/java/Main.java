import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.Optimizers.AdaptiveMomentum;
import NeuralNetwork.Optimizers.OptimizerBase;
import Utilities.Factory;
import Utilities.Helper;

public class Main {
    public static void main(String[] args) {
        final long startTime = System.currentTimeMillis();

        final Batch inputBatch = Factory.getInputBatch();
        final Batch targetBatch = Factory.getTargetBatch();

        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork();
        final OptimizerBase optimizer =
                new AdaptiveMomentum(neuralNetwork, 0.05, 0.00005, 0.0000001, 0.9, 0.999);

        for (int i = 0; i < 2_501; ++i) {
            boolean printing = i % 100 == 0;

            neuralNetwork.forward(inputBatch, targetBatch, true);

            if (printing) {
                final double accuracy = neuralNetwork.getAccuracy();
                final double loss = neuralNetwork.getLoss();
                final double regularizedLoss = neuralNetwork.getRegularizedLoss();
                System.out.printf(
                        "epoch: %-15d accuracy: %-15s loss: %-15s regularized loss: %-15s",
                        i,
                        Helper.formatNumber(accuracy, 5),
                        Helper.formatNumber(loss, 5),
                        Helper.formatNumber(regularizedLoss, 5)
                );
            }

            neuralNetwork.backward();
            optimizer.performOptimization();
            neuralNetwork.clearState();

            if (printing) {
                final double learningRate = optimizer.getCurrentLearningRate();
                System.out.printf("%-15s%n", "lr: " + learningRate);
            }
         }

        final long endTime = System.currentTimeMillis();
        System.out.println("Time needed: " + (endTime - startTime) + " ms");


        final Batch testInputBatch = Factory.getTestingInputBatch();
        final Batch testTargetBatch = Factory.getTestingTargetBatch();

        neuralNetwork.forward(testInputBatch, testTargetBatch, false);

        final double accuracy = neuralNetwork.getAccuracy();
        final double loss = neuralNetwork.getLoss();
        System.out.print("accuracy: " + accuracy + ", loss: " + loss);
    }
}
