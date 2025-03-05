import GUI.GUI;
import NeuralNetwork.BuildingBlocks.Batch;
import NeuralNetwork.NeuralNetwork;
import NeuralNetwork.Optimizers.AdaptiveMomentum;
import NeuralNetwork.Optimizers.OptimizerBase;
import Utilities.Factory;
import Utilities.Helper;

public class Main {
    public static void main(String[] args) {
        final GUI gui = new GUI();

        final long startTime = System.currentTimeMillis();


        final Batch trainInputBatch = Factory.getTrainInputBatch();
        final Batch trainTargetBatch = Factory.getTargetBatch();

        final Batch testInputBatch = Factory.getTestInputBatch();
        final Batch testTargetBatch = Factory.getTestTargetBatch();


        final NeuralNetwork neuralNetwork = Factory.getNeuralNetwork();
        final OptimizerBase optimizer =
                new AdaptiveMomentum(neuralNetwork, 0.01, 0.001, 0.0000001, 0.9, 0.999);

        for (int i = 0; i < 551; ++i) {
            boolean printing = i % 50 == 0;

            neuralNetwork.forward(trainInputBatch, trainTargetBatch, true);

            if (printing) {
                final double accuracy = neuralNetwork.getAccuracyForPrinting();
                final double loss = neuralNetwork.getLossForPrinting();
                final double regularizedLoss = neuralNetwork.getRegularizedLossForPrinting();
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


        neuralNetwork.forward(testInputBatch, testTargetBatch, true);

        final Batch predictions = neuralNetwork.getLayers().get(6).getSavedInputBatch();
        final Batch targets = neuralNetwork.getLayers().get(6).getSavedTargetBatch();

        gui.show(predictions.getColumn(0), targets.getColumn(0));

        /* final Batch testInputBatch = Factory.getTestingInputBatch();
        final Batch testTargetBatch = Factory.getTestingTargetBatch();

        neuralNetwork.forward(testInputBatch, testTargetBatch, false);

        final double accuracy = neuralNetwork.getAccuracyForPrinting();
        final double loss = neuralNetwork.getLossForPrinting();

        final Batch predictions = neuralNetwork.getLayers().get(6).getSavedInputBatch();
        final Batch targets = neuralNetwork.getLayers().get(6).getSavedTargetBatch();

        System.out.print("{\n");
        for (int i = 0; i < predictions.getRowsSize(); ++i) {
            final double prediction = predictions.getRow(i).getValue(0);
            final double target = targets.getRow(i).getValue(0);

            System.out.print("\t[ " + target + ", " + prediction + " ]\n");
        }
        System.out.print("}\n\n");

        System.out.print("accuracy: " + accuracy + ", loss: " + loss); */
    }
}
