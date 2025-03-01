import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.Optimizers.StochasticGradientDescent;
import Utilities.Factory;

public class Main {
    public static void main(String[] args) {
        final Batch inputBatch = Factory.getInputBatch();
        final Batch targetBatch = Factory.getTargetBatch();

        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork();
        final StochasticGradientDescent optimizer = new StochasticGradientDescent(neuralNetwork, 1, 1);

        for (int i = 0; i < 100_000; ++i) {
            neuralNetwork.forward(inputBatch, targetBatch);

            if (i % 1_000 == 0) {
                final double loss = neuralNetwork.getLoss();
                final double accuracy = neuralNetwork.getAccuracy();
                System.out.println("epoch: "  + i + ", accuracy: " + accuracy + ", loss: " + loss);
            }

            neuralNetwork.backward();
            optimizer.optimize();
            neuralNetwork.clearState();
        }
    }
}
