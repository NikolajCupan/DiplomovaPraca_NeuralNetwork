package NeuralNetwork;

import NeuralNetwork.ActivationFunctions.*;
import NeuralNetwork.BuildingBlocks.DataList;
import NeuralNetwork.BuildingBlocks.Neuron;
import NeuralNetwork.BuildingBlocks.RegularizerStruct;
import NeuralNetwork.Layers.Common.ActivationLayer;
import NeuralNetwork.Layers.Common.DropoutLayer;
import NeuralNetwork.Layers.Common.HiddenLayer;
import NeuralNetwork.Layers.Common.LossLayer;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.Layers.Special.SoftmaxCategoricalCrossEntropyLayer;
import NeuralNetwork.LossFunctions.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Objects;

public class NeuralNetworkHelper {
    private static final String LINEAR_ACTIVATION_FUNCTION_NAME = "LINEAR";
    private static final String RECTIFIED_LINEAR_UNIT_ACTIVATION_FUNCTION_NAME = "RECTIFIED_LINEAR_UNIT";
    private static final String SIGMOID_ACTIVATION_FUNCTION_NAME = "SIGMOID";
    private static final String SOFTMAX_ACTIVATION_FUNCTION_NAME = "SOFTMAX";


    private static final String BINARY_CROSS_ENTROPY_LOSS_FUNCTION_NAME = "BINARY_CROSS_ENTROPY";
    private static final String CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION_NAME = "CATEGORICAL_CROSS_ENTROPY";

    private static final String REGRESSION_LOSS_FUNCTION_NAME = "REGRESSION_LOSS_FUNCTION";
    private static final String MEAN_ABSOLUTE_ERROR_LOSS_FUNCTION_NAME = "MEAN_ABSOLUTE_ERROR";
    private static final String MEAN_SQUARED_ERROR_LOSS_FUNCTION_NAME = "MEAN_SQUARED_ERROR";


    private static final String HIDDEN_LAYER_NAME = "HIDDEN_LAYER";
    private static final String DROPOUT_LAYER_NAME = "DROPOUT_LAYER";
    private static final String ACTIVATION_LAYER_NAME = "ACTIVATION_LAYER";
    private static final String LOSS_LAYER_NAME = "LOSS_LAYER";
    private static final String SOFTMAX_CROSS_CATEGORICAL_ENTROPY_LAYER_NAME = "SOFTMAX_CATEGORICAL_CROSS_ENTROPY_LAYER";


    private static final String GLOBAL_REGULARIZER_NAME = "GLOBAL_REGULARIZER";
    private static final String LAYER_REGULARIZER_NAME = "LAYER_REGULARIZER";


    public static void saveToFile(final NeuralNetwork neuralNetwork, final String fileName) {
        try {
            final FileOutputStream fileOutputStream = new FileOutputStream(fileName);
            final OutputStreamWriter outputStreamWriter = new OutputStreamWriter(fileOutputStream, StandardCharsets.UTF_8);
            final Writer writer = new BufferedWriter(outputStreamWriter);


            final List<LayerBase> layers = neuralNetwork.getLayers();
            assert(!layers.isEmpty());

            final HiddenLayer firstLayer = (HiddenLayer)layers.getFirst();
            final int inputsSize = firstLayer.getWeightsSize();
            writer.write(String.valueOf(inputsSize));


            if (neuralNetwork.isGlobalRegularizerSet()) {
                final RegularizerStruct regularizer = neuralNetwork.getGlobalRegularizer();

                writer.write("\n" + NeuralNetworkHelper.GLOBAL_REGULARIZER_NAME + "\n");
                writer.write(
                        regularizer.getBiasesRegularizerL1() + "," +
                        regularizer.getBiasesRegularizerL2() + "," +
                        regularizer.getWeightsRegularizerL1() + "," +
                        regularizer.getWeightsRegularizerL2()
                );
            }


            for (final LayerBase layer : layers) {
                 writer.write("\n");

                switch (layer) {
                    case final HiddenLayer hiddenLayer -> NeuralNetworkHelper.saveHiddenLayer(neuralNetwork, hiddenLayer, writer);
                    case final DropoutLayer dropoutLayer -> NeuralNetworkHelper.saveDropoutLayer(dropoutLayer, writer);
                    case final ActivationLayer activationLayer -> NeuralNetworkHelper.saveActivationLayer(activationLayer, writer);
                    case final LossLayer lossLayer -> NeuralNetworkHelper.saveLossLayer(lossLayer, writer);
                    case final SoftmaxCategoricalCrossEntropyLayer softmaxCCELayer -> NeuralNetworkHelper.saveSoftmaxCategoricalCrossEntropyLayer(softmaxCCELayer, writer);
                    case null, default -> throw new RuntimeException("Unknown layer class");
                }
            }


            writer.close();
            outputStreamWriter.close();
            fileOutputStream.close();
        } catch (final Exception exception) {
            System.out.println("Neural network could not be saved to file");
            System.out.println(exception.getMessage());
        }
    }

    private static void saveHiddenLayer(final NeuralNetwork neuralNetwork, final HiddenLayer hiddenLayer, final Writer writer) throws Exception {
        writer.write(NeuralNetworkHelper.HIDDEN_LAYER_NAME + "\n");

        if (!neuralNetwork.isGlobalRegularizerSet() && hiddenLayer.isRegularizerPresent()) {
            final RegularizerStruct regularizer = hiddenLayer.getRegularizer();

            writer.write(NeuralNetworkHelper.LAYER_REGULARIZER_NAME + "\n");
            writer.write(
                    regularizer.getBiasesRegularizerL1() + "," +
                    regularizer.getBiasesRegularizerL2() + "," +
                    regularizer.getWeightsRegularizerL1() + "," +
                    regularizer.getWeightsRegularizerL2() + "\n"
            );
        }

        writer.write(hiddenLayer.getNeuronsSize() + "\n");
        writer.write(String.valueOf(hiddenLayer.getWeightsSize()));

        final List<Neuron> neurons = hiddenLayer.getNeurons();
        for (final Neuron neuron : neurons) {
            writer.write("\n");
            writer.write(String.valueOf(neuron.getBias()));

            final double[] weights = neuron.getWeights().getDataListRawValues();
            for (final double weight : weights) {
                writer.write("," + weight);
            }
        }
    }

    private static void saveDropoutLayer(final DropoutLayer dropoutLayer, final Writer writer) throws Exception {
        writer.write(NeuralNetworkHelper.DROPOUT_LAYER_NAME + "\n");
        writer.write(String.valueOf(dropoutLayer.getKeepRate()));
    }

    private static void saveActivationLayer(final ActivationLayer activationLayer, final Writer writer) throws Exception {
        writer.write(NeuralNetworkHelper.ACTIVATION_LAYER_NAME + "\n");
        final IActivationFunction activationFunction = activationLayer.getActivationFunction();

        switch (activationFunction) {
            case final Linear ignore -> writer.write(NeuralNetworkHelper.LINEAR_ACTIVATION_FUNCTION_NAME);
            case final RectifiedLinearUnit ignore -> writer.write(NeuralNetworkHelper.RECTIFIED_LINEAR_UNIT_ACTIVATION_FUNCTION_NAME);
            case final Sigmoid ignore -> writer.write(NeuralNetworkHelper.SIGMOID_ACTIVATION_FUNCTION_NAME);
            case final Softmax ignore -> writer.write(NeuralNetworkHelper.SOFTMAX_ACTIVATION_FUNCTION_NAME);
            case null, default -> throw new RuntimeException("Unknown activation function class");
        }
    }

    private static void saveLossLayer(final LossLayer lossLayer, final Writer writer) throws Exception {
        writer.write(NeuralNetworkHelper.LOSS_LAYER_NAME + "\n");
        final ILossFunction lossFunction = lossLayer.getLossFunction();

        switch (lossFunction) {
            case final BinaryCrossEntropy ignore -> writer.write(NeuralNetworkHelper.BINARY_CROSS_ENTROPY_LOSS_FUNCTION_NAME);
            case final CategoricalCrossEntropy ignore -> writer.write(NeuralNetworkHelper.CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION_NAME);
            case final RegressionLossFunction regressionLossFunction -> {
                writer.write(NeuralNetworkHelper.REGRESSION_LOSS_FUNCTION_NAME + "\n");

                if (lossFunction instanceof MeanAbsoluteError) {
                    writer.write(NeuralNetworkHelper.MEAN_ABSOLUTE_ERROR_LOSS_FUNCTION_NAME + "\n");
                } else if (lossFunction instanceof MeanSquaredError) {
                    writer.write(NeuralNetworkHelper.MEAN_SQUARED_ERROR_LOSS_FUNCTION_NAME + "\n");
                } else {
                    throw new RuntimeException("Unknown regression loss function class");
                }

                writer.write(String.valueOf(regressionLossFunction.getMaxPercentageDifference()));
            }
            case null, default -> throw new RuntimeException("Unknown loss function class");
        }
    }

    private static void saveSoftmaxCategoricalCrossEntropyLayer(final SoftmaxCategoricalCrossEntropyLayer ignore, final Writer writer) throws Exception {
        writer.write(NeuralNetworkHelper.SOFTMAX_CROSS_CATEGORICAL_ENTROPY_LAYER_NAME);
    }

    public static NeuralNetwork loadFromFile(final String fileName) {
        try {
            final FileReader fileReader = new FileReader(fileName);
            final BufferedReader bufferedReader = new BufferedReader(fileReader);


            String line = bufferedReader.readLine();
            final int inputsSize = Integer.parseInt(line);
            final NeuralNetwork neuralNetwork = new NeuralNetwork(inputsSize);

            while ((line = bufferedReader.readLine()) != null) {
                switch (line) {
                    case NeuralNetworkHelper.GLOBAL_REGULARIZER_NAME -> NeuralNetworkHelper.loadGlobalRegularizer(neuralNetwork, bufferedReader);
                    case NeuralNetworkHelper.HIDDEN_LAYER_NAME -> NeuralNetworkHelper.loadHiddenLayer(neuralNetwork, bufferedReader);
                    case NeuralNetworkHelper.DROPOUT_LAYER_NAME -> NeuralNetworkHelper.loadDropoutLayer(neuralNetwork, bufferedReader);
                    case NeuralNetworkHelper.ACTIVATION_LAYER_NAME -> NeuralNetworkHelper.loadActivationLayer(neuralNetwork, bufferedReader);
                    case NeuralNetworkHelper.LOSS_LAYER_NAME -> NeuralNetworkHelper.loadLossLayer(neuralNetwork, bufferedReader);
                    case NeuralNetworkHelper.SOFTMAX_CROSS_CATEGORICAL_ENTROPY_LAYER_NAME -> NeuralNetworkHelper.loadSoftmaxCategoricalCrossEntropyLayer(neuralNetwork, bufferedReader);
                    case null, default -> throw new RuntimeException("Unknown layer class");
                }
            }


            bufferedReader.close();
            fileReader.close();

            return neuralNetwork;
        } catch (final Exception exception) {
            System.out.println("Neural network could not be loaded from file");
            System.out.println(exception.getMessage());
        }

        return null;
    }

    private static void loadGlobalRegularizer(final NeuralNetwork neuralNetwork, final BufferedReader bufferedReader) throws Exception {
        final String line = bufferedReader.readLine();
        final String[] parts = line.split(",");

        final RegularizerStruct globalRegularizer = new RegularizerStruct(
                Double.parseDouble(parts[0]),
                Double.parseDouble(parts[1]),
                Double.parseDouble(parts[2]),
                Double.parseDouble(parts[3])
        );

        neuralNetwork.initializeGlobalRegularizer(globalRegularizer);
    }

    private static void loadHiddenLayer(final NeuralNetwork neuralNetwork, final BufferedReader bufferedReader) throws Exception {
        final HiddenLayer hiddenLayer = new HiddenLayer();

        String line = bufferedReader.readLine();
        if (Objects.equals(line, NeuralNetworkHelper.LAYER_REGULARIZER_NAME)) {
            final String regularizerLine = bufferedReader.readLine();
            final String[] parts = regularizerLine.split(",");

            final RegularizerStruct regularizer = new RegularizerStruct(
                    Double.parseDouble(parts[0]),
                    Double.parseDouble(parts[1]),
                    Double.parseDouble(parts[2]),
                    Double.parseDouble(parts[3])
            );

            hiddenLayer.initializeRegularizer(regularizer);
            line = bufferedReader.readLine();
        }

        final int neuronsSize = Integer.parseInt(line);
        final int weightsSize = Integer.parseInt(bufferedReader.readLine());



        for (int neuronIndex = 0; neuronIndex < neuronsSize; ++neuronIndex) {
            final String neuronLine = bufferedReader.readLine();
            final String[] parts = neuronLine.split(",");

            final double bias = Double.parseDouble(parts[0]);
            final DataList weights = new DataList(weightsSize);

            for (int i = 1; i < parts.length; ++i) {
                weights.setValue(i - 1, Double.parseDouble(parts[i]));
            }

            hiddenLayer.addNeuron(new Neuron(bias, weights));
        }

        neuralNetwork.addHiddenLayer(hiddenLayer);
    }

    private static void loadDropoutLayer(final NeuralNetwork neuralNetwork, final BufferedReader bufferedReader) throws Exception {
        final double keptRate = Double.parseDouble(bufferedReader.readLine());
        neuralNetwork.addDropoutLayer(new DropoutLayer(keptRate));
    }

    private static void loadActivationLayer(final NeuralNetwork neuralNetwork, final BufferedReader bufferedReader) throws Exception {
        final String activationFunctionName = bufferedReader.readLine();
        IActivationFunction activationFunction;

        switch (activationFunctionName) {
            case NeuralNetworkHelper.LINEAR_ACTIVATION_FUNCTION_NAME -> activationFunction = new Linear();
            case NeuralNetworkHelper.RECTIFIED_LINEAR_UNIT_ACTIVATION_FUNCTION_NAME -> activationFunction = new RectifiedLinearUnit();
            case NeuralNetworkHelper.SIGMOID_ACTIVATION_FUNCTION_NAME -> activationFunction = new Sigmoid();
            case NeuralNetworkHelper.SOFTMAX_ACTIVATION_FUNCTION_NAME -> activationFunction = new Softmax();
            case null, default -> throw new RuntimeException("Unknown activation function class");
        }

        neuralNetwork.addActivationLayer(new ActivationLayer(activationFunction));
    }

    private static void loadLossLayer(final NeuralNetwork neuralNetwork, final BufferedReader bufferedReader) throws Exception {
        final String lossFunctionName = bufferedReader.readLine();
        ILossFunction lossFunction;

        switch (lossFunctionName) {
            case NeuralNetworkHelper.BINARY_CROSS_ENTROPY_LOSS_FUNCTION_NAME -> lossFunction = new BinaryCrossEntropy();
            case NeuralNetworkHelper.CATEGORICAL_CROSS_ENTROPY_LOSS_FUNCTION_NAME -> lossFunction = new CategoricalCrossEntropy();
            case NeuralNetworkHelper.REGRESSION_LOSS_FUNCTION_NAME -> {
                final String regressionLossFunctionName = bufferedReader.readLine();
                final double maxPercentageDifference = Double.parseDouble(bufferedReader.readLine());

                if (Objects.equals(regressionLossFunctionName, NeuralNetworkHelper.MEAN_ABSOLUTE_ERROR_LOSS_FUNCTION_NAME)) {
                    lossFunction = new MeanAbsoluteError(maxPercentageDifference);
                } else if (Objects.equals(regressionLossFunctionName, NeuralNetworkHelper.MEAN_SQUARED_ERROR_LOSS_FUNCTION_NAME)) {
                    lossFunction = new MeanSquaredError(maxPercentageDifference);
                } else {
                    throw new RuntimeException("Unknown regression loss function class");
                }
            }
            case null, default -> throw new RuntimeException("Unknown loss function class");
        }

        neuralNetwork.addLossLayer(new LossLayer(lossFunction));
    }

    private static void loadSoftmaxCategoricalCrossEntropyLayer(final NeuralNetwork neuralNetwork, final BufferedReader bufferedReader) {
        neuralNetwork.addSpecialLayer(new SoftmaxCategoricalCrossEntropyLayer());
    }
}
