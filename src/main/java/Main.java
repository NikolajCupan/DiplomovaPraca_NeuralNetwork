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

        for (int i = 0; i < 10_000; ++i) {
            neuralNetwork.forward(inputBatch, targetBatch);

            if (i % 1_000 == 0) {
                System.out.println(i + ". " + neuralNetwork.getAccuracy());
            }

            neuralNetwork.backward();
            optimizer.optimize();
            neuralNetwork.clearState();
        }

        int x = 100;
    }
}
