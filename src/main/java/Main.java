import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.Optimizers.StochasticGradientDescentWithMomentum;
import Utilities.Factory;

public class Main {
    public static void main(String[] args) {
        final long startTime = System.currentTimeMillis();

        final Batch inputBatch = Factory.getInputBatch();
        final Batch targetBatch = Factory.getTargetBatch();

        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork();
        final StochasticGradientDescentWithMomentum optimizer =
                new StochasticGradientDescentWithMomentum(neuralNetwork, 1.0, 0.001, 0.9);

        for (int i = 0; i < 2_501; ++i) {
            boolean printing = i % 500 == 0;

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
