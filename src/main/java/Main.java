import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.Optimizers.StochasticGradientDescent;
import Utilities.Factory;

public class Main {
    public static void main(String[] args) {
//        final double startingLearningRate = 1.0;
//        final double learningRateDecay = 0.1;
//        final int stepsSize = 10_000;
//        final int printEach = 100;
//
//        for (int step = 0; step <= stepsSize; ++step) {
//            if (step % printEach == 0) {
//                final double learningRate =
//                        startingLearningRate * (1 / (1 + learningRateDecay * step));
//                System.out.println(step + " " + learningRate);
//            }
//        }
//
//        if (1 == 1) {
//            return;
//        }

        final long startTime = System.currentTimeMillis();

        final Batch inputBatch = Factory.getInputBatch();
        final Batch targetBatch = Factory.getTargetBatch();

        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork();
        final StochasticGradientDescent optimizer = new StochasticGradientDescent(neuralNetwork, 1.0, 0.001);

        for (int i = 0; i < 50_000; ++i) {
            boolean printing = false;
            if (i % 1_000 == 0) {
                printing = true;
            }

            neuralNetwork.forward(inputBatch, targetBatch);

            if (printing) {
                final double accuracy = neuralNetwork.getAccuracy();
                final double loss = neuralNetwork.getLoss();
                System.out.print("epoch: "  + i + ", accuracy: " + accuracy + ", loss: " + loss);
            }

            neuralNetwork.backward();
            optimizer.optimize();
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
