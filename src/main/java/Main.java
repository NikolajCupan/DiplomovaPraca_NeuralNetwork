import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.Optimizers.AdaptiveGradient;
import Utilities.Factory;

public class Main {
    public static void main(String[] args) {
        final long startTime = System.currentTimeMillis();

        final Batch inputBatch = Factory.getInputBatch();
        final Batch targetBatch = Factory.getTargetBatch();

        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork();
        final AdaptiveGradient optimizer =
                new AdaptiveGradient(neuralNetwork, 1.0, 0.0001, 0.0000001);

        for (int i = 0; i < 10001; ++i) {
            boolean printing = i % 1000 == 0;

            neuralNetwork.forward(inputBatch, targetBatch);

            if (printing) {
                final double accuracy = neuralNetwork.getAccuracy();
                final double loss = neuralNetwork.getLoss();
                System.out.print("epoch: "  + i + ", accuracy: " + accuracy + ", loss: " + loss);
            }

            neuralNetwork.backward();
            optimizer.performOptimization();
            neuralNetwork.clearState();

            if (printing) {
                final double learningRate = optimizer.getCurrentLearningRate();
                System.out.println(", lr: " + learningRate);
            }
         }

        final long endTime = System.currentTimeMillis();
        System.out.println("Time needed: " + (endTime - startTime) + " ms");
    }
}
