package NeuralNetwork;

import NeuralNetwork.BuildingBlocks.*;
import NeuralNetwork.Layers.Common.ActivationLayer;
import NeuralNetwork.Layers.Common.DropoutLayer;
import NeuralNetwork.Layers.Common.HiddenLayer;
import NeuralNetwork.Layers.Common.LossLayer;
import NeuralNetwork.Layers.IAccuracyForPrintingGetter;
import NeuralNetwork.Layers.ILossForPrintingGetter;
import NeuralNetwork.Layers.LayerBase;
import NeuralNetwork.Layers.Special.SoftmaxCategoricalCrossEntropyLayer;
import NeuralNetwork.Optimizers.OptimizerBase;
import org.json.JSONObject;

import java.io.PrintStream;
import java.util.*;

public class NeuralNetwork {
    private volatile boolean stopTraining;
    private final PrintStream outputStream;

    private final int inputSize;
    private final List<LayerBase> layers;

    private Optional<RegularizerStruct> globalRegularizerStruct;
    private Optional<OptimizerBase> optimizer;

    public NeuralNetwork(final int inputSize, final PrintStream outputStream) {
        this.stopTraining = false;
        this.outputStream = outputStream;

        this.inputSize = inputSize;
        this.layers = new ArrayList<>();

        this.globalRegularizerStruct = Optional.empty();
        this.optimizer = Optional.empty();
    }

    public void initializeGlobalRegularizer(final RegularizerStruct regularizerStruct) {
        if (this.globalRegularizerStruct.isPresent()) {
            throw new RuntimeException("Global regularizer is already set in neural network");
        }

        if (!this.layers.isEmpty()) {
            throw new RuntimeException("Global regularizer cannot be set if layers are not empty");
        }

        this.globalRegularizerStruct = Optional.of(regularizerStruct);
    }

    public boolean isGlobalRegularizerSet() {
        return this.globalRegularizerStruct.isPresent();
    }

    public RegularizerStruct getGlobalRegularizer() {
        if (this.globalRegularizerStruct.isEmpty()) {
            throw new RuntimeException("Global regularizer is empty");
        }

        return this.globalRegularizerStruct.get();
    }

    private double getAccuracyForPrinting() {
        final LayerBase lastLayer = this.layers.getLast();

        if (lastLayer instanceof final IAccuracyForPrintingGetter lossLayer) {
            return lossLayer.getAccuracyForPrinting();
        } else {
            throw new RuntimeException("Last layer cannot be used to calculate accuracy");
        }
    }

    private double getLossForPrinting() {
        final LayerBase lastLayer = this.layers.getLast();

        if (lastLayer instanceof final ILossForPrintingGetter lossLayer) {
            return lossLayer.getLossForPrinting();
        } else {
            throw new RuntimeException("Last layer cannot be used to calculate loss");
        }
    }

    private double getRegularizedLossForPrinting() {
        double regularizedLoss = this.getLossForPrinting();

        for (final LayerBase layer : this.layers) {
            if (layer instanceof final HiddenLayer hiddenLayer && hiddenLayer.isRegularizerPresent()) {
                regularizedLoss += hiddenLayer.getRegularizedLoss();
            }
        }

        return regularizedLoss;
    }

    public Batch getPredictedBatch() {
        final LayerBase lastLayer = this.layers.getLast();

        if (lastLayer instanceof LossLayer) {
            return lastLayer.getSavedInputBatch();
        } else if (lastLayer instanceof final SoftmaxCategoricalCrossEntropyLayer softmaxCCELayer) {
            return softmaxCCELayer.getLossLayer().getSavedInputBatch();
        } else {
            throw new RuntimeException("Last layer cannot be used to get predictions");
        }
    }

    public List<LayerBase> getLayers() {
        return this.layers;
    }

    public void stopTraining() {
        final JSONObject json = new JSONObject();
        json.put("action", "training_stopped");
        json.put("reason", "requested");
        this.outputStream.println(json);

        this.stopTraining = true;
    }

    public void train(final Batch inputBatch, final Batch targetBatch, final int epochsSize, final int stepRowsSize, final int epochPrintEvery, final int stepPrintEvery, final int timeLimitMs) {
        this.stopTraining = false;

        try {
            final Thread trainingThread = new Thread(
                    () -> this.train(inputBatch, targetBatch, epochsSize, stepRowsSize, epochPrintEvery, stepPrintEvery)
            );
            trainingThread.start();

            final long startTime = System.currentTimeMillis();
            while (trainingThread.isAlive()) {
                final long elapsedTime = System.currentTimeMillis() - startTime;
                if (elapsedTime > timeLimitMs) {
                    final JSONObject json = new JSONObject();
                    json.put("action", "training_stopped");
                    json.put("reason", "timeout");
                    this.outputStream.println(json);

                    this.stopTraining = true;
                    break;
                }

                Thread.sleep(500);
            }

            trainingThread.join();
        } catch (final Exception ignore) {
        }

        this.stopTraining = false;
    }

    public void train(final Batch inputBatch, final Batch targetBatch, final int epochsSize, final int stepRowsSize, final int epochPrintEvery, final int stepPrintEvery) {
        if (!this.isLastLayerSuitable()) {
            throw new RuntimeException("Cannot perform forward method, last layer is not suitable");
        } else if (this.optimizer.isEmpty()) {
            throw new RuntimeException("Cannot perform training, optimizer is empty");
        }

        this.clearState();
        final List<Batch> inputBatchSteps = NeuralNetwork.prepareSteps(inputBatch, stepRowsSize);
        final List<Batch> targetBatchSteps = NeuralNetwork.prepareSteps(targetBatch, stepRowsSize);

        for (int epoch = 1; epoch < epochsSize + 1; ++epoch) {
            if (this.stopTraining) {
                this.outputStream.println(new JSONObject().put("message", "Training stopped at epoch: " + epoch));
                break;
            }


            final boolean printingEpoch = (epoch % epochPrintEvery == 0);

            double sumAccuracy = 0.0;
            double sumLoss = 0.0;
            double sumRegularizedLoss = 0.0;
            double sumLearningRate = 0.0;


            for (int stepIndex = 0; stepIndex < inputBatchSteps.size(); ++stepIndex) {
                final boolean printingStep = ((stepIndex + 1) % stepPrintEvery == 0);
                final JSONObject stepJson = new JSONObject();

                final Batch stepInputBatch = inputBatchSteps.get(stepIndex);
                final Batch targetInputBatch = targetBatchSteps.get(stepIndex);

                this.forward(stepInputBatch, targetInputBatch, true);


                if (printingEpoch || printingStep) {
                    final double accuracy = this.getAccuracyForPrinting();
                    sumAccuracy += accuracy;

                    final double loss = this.getLossForPrinting();
                    sumLoss += loss;

                    final double regularizedLoss = this.getRegularizedLossForPrinting();
                    sumRegularizedLoss += regularizedLoss;

                    if (printingEpoch && printingStep) {
                        stepJson.put("epoch", epoch);
                        stepJson.put("step", stepIndex);
                        stepJson.put("step_accuracy", accuracy);
                        stepJson.put("step_loss", loss);
                        stepJson.put("step_regularized_loss", regularizedLoss);
                    }
                }


                this.backward();
                this.optimizer.get().performOptimization();
                this.clearState();


                if (printingEpoch || printingStep) {
                    final double learningRate = this.optimizer.get().getCurrentLearningRate();
                    sumLearningRate += learningRate;

                    if (printingEpoch && printingStep) {
                        stepJson.put("learning_rate", learningRate);
                        this.outputStream.println(stepJson);
                    }
                }
            }

            if (printingEpoch) {
                final JSONObject epochJson = new JSONObject();
                epochJson.put("epoch", epoch);
                epochJson.put("accuracy", sumAccuracy / inputBatchSteps.size());
                epochJson.put("loss", sumLoss / inputBatchSteps.size());
                epochJson.put("regularized_loss", sumRegularizedLoss / inputBatchSteps.size());
                epochJson.put("learning_rate", sumLearningRate / inputBatchSteps.size());
                this.outputStream.println(epochJson);
            }
        }

        this.clearState();
    }

    public void test(final Batch inputBatch, final Batch targetBatch) {
        this.clearState();
        this.forward(inputBatch, targetBatch, false);

        final double accuracy = this.getAccuracyForPrinting();
        final double loss = this.getLossForPrinting();

        final JSONObject json = new JSONObject();
        json.put("accuracy", accuracy);
        json.put("loss", loss);
        this.outputStream.println(json);
    }

    private void forward(final Batch inputBatch, final Batch targetBatch, final boolean includeDropoutLayers) {
        final List<LayerBase> usedLayers = new ArrayList<>();
        for (final LayerBase layer : this.layers) {
            if (includeDropoutLayers || !(layer instanceof DropoutLayer)) {
                usedLayers.add(layer);
            }
        }

        final LayerBase lastLayer = usedLayers.getLast();
        lastLayer.setSavedTargetBatch(targetBatch);

        NeuralNetwork.forwardLayers(inputBatch, usedLayers);
    }

    public Batch predict(final Batch inputBatch) {
        this.clearState();

        final List<LayerBase> usedLayers = NeuralNetwork.filterLayersForPrediction(this.layers);
        NeuralNetwork.forwardLayers(inputBatch, usedLayers);

        final Batch predictedBatch = usedLayers.getLast().getSavedOutputBatch();
        this.clearState();

        return predictedBatch;
    }

    public DataList forecast(final DataList startList, final int forecastsSize) {
        this.clearState();

        final List<LayerBase> usedLayers = NeuralNetwork.filterLayersForPrediction(this.layers);
        final DataList predictions = new DataList(forecastsSize);

        final DataList currentList = new DataList(startList.getDataListSize());
        for (int i = 0; i < currentList.getDataListSize(); ++i) {
            currentList.setValue(i, startList.getValue(i));
        }

        for (int i = 0; i < forecastsSize; ++i) {
            final Batch batch = new Batch();
            batch.addRow(currentList);


            NeuralNetwork.forwardLayers(batch, usedLayers);
            final double predictedValue = usedLayers.getLast().getSavedInputBatch().getRow(0).getValue(0);
            predictions.setValue(i, predictedValue);


            for (int j = 0; j < currentList.getDataListSize() - 1; ++j) {
                currentList.setValue(
                        j,
                        currentList.getValue(j + 1)
                );
            }
            currentList.setValue(currentList.getDataListSize() - 1, predictedValue);

            this.clearState();
        }

        return predictions;
    }

    private static void forwardLayers(final Batch inputBatch, final List<LayerBase> layers) {
        final HiddenLayer firstLayer = NeuralNetwork.getLayerAsType(layers, 0, HiddenLayer.class);
        firstLayer.forward(inputBatch);

        for (int i = 1; i < layers.size(); ++i) {
            final LayerBase previousLayer = layers.get(i - 1);
            final LayerBase currentLayer = layers.get(i);

            currentLayer.forward(previousLayer.getSavedOutputBatch());
        }
    }

    private static List<LayerBase> filterLayersForPrediction(final List<LayerBase> layers) {
        final List<LayerBase> filteredLayers = new ArrayList<>();

        for (final LayerBase layer : layers) {
            if (layer instanceof final SoftmaxCategoricalCrossEntropyLayer softmaxCCELayer) {
                filteredLayers.add(softmaxCCELayer.getActivationLayer());
            } else if (!(layer instanceof DropoutLayer) && !(layer instanceof LossLayer)) {
                filteredLayers.add(layer);
            }
        }

        return filteredLayers;
    }

    private void backward() {
        final LayerBase lastLayer = this.layers.getLast();
        lastLayer.backward(new GradientStruct());

        for (int i = this.layers.size() - 1; i > 0; --i) {
            final LayerBase previousLayer = this.layers.get(i - 1);
            final LayerBase currentLayer = this.layers.get(i);

            previousLayer.backward(currentLayer.getSavedOutputGradientStruct());
        }
    }

    private void clearState() {
        for (final LayerBase layer : this.layers) {
            layer.clearState();
        }
    }

    public void addHiddenLayer(final HiddenLayer hiddenLayerToBeAdded) {
        if (this.layers.isEmpty()) {
            final int firstLayerWeightsSize = hiddenLayerToBeAdded.getWeightsSize();

            if (this.inputSize != firstLayerWeightsSize) {
                throw new IllegalArgumentException("Input size [" + this.inputSize + "] is not equal to weights size of first layer [" + firstLayerWeightsSize + "]");
            }
        } else {
            final LayerBase lastLayer = this.layers.getLast();
            if (!hiddenLayerToBeAdded.isCompatible(lastLayer)) {
                throw new IllegalArgumentException("Hidden layer cannot be placed after " + lastLayer.getClass());
            }
            assert(this.layers.size() >= 2);

            final LayerBase penultimateLayer = this.layers.get(this.layers.size() - 2);

            if (penultimateLayer instanceof final HiddenLayer penultimateHiddenLayer) {
                final int penultimateLayerNeuronsSize = penultimateHiddenLayer.getNeuronsSize();
                final int hiddenLayerToBeAddedWeightsSize = hiddenLayerToBeAdded.getWeightsSize();

                if (penultimateLayerNeuronsSize != hiddenLayerToBeAddedWeightsSize) {
                    throw new IllegalArgumentException(
                            "Current last hidden layer neurons size [" + penultimateLayerNeuronsSize + "] is not equal to new hidden layer weights size [" + hiddenLayerToBeAddedWeightsSize + "]"
                    );
                }
            }
        }

        if (this.globalRegularizerStruct.isPresent()) {
            try {
                hiddenLayerToBeAdded.initializeRegularizer(this.globalRegularizerStruct.get());
            } catch (final Exception IllegalArgumentException) {
                throw new IllegalArgumentException("Neural network could not set regularizer for hidden layer, hidden layer already has regularizer set");
            }
        }

        this.layers.add(hiddenLayerToBeAdded);
    }

    public void addActivationLayer(final ActivationLayer activationLayerToBeAdded) {
        if (this.layers.isEmpty()) {
            throw new IllegalArgumentException("Activation layer cannot be the first layer in neural network");
        }

        final LayerBase lastLayer = this.layers.getLast();
        if (!activationLayerToBeAdded.isCompatible(lastLayer)) {
            throw new IllegalArgumentException("Activation layer cannot be placed after " + lastLayer.getClass());
        }

        this.layers.add(activationLayerToBeAdded);
    }

    public void addDropoutLayer(final DropoutLayer dropoutLayerToBeAdded) {
        if (this.layers.isEmpty()) {
            throw new IllegalArgumentException("Dropout layer cannot be the first layer in neural network");
        }

        final LayerBase lastLayer = this.layers.getLast();
        if (!dropoutLayerToBeAdded.isCompatible(lastLayer)) {
            throw new IllegalArgumentException("Dropout layer cannot be placed after " + lastLayer.getClass());
        }

        this.layers.add(dropoutLayerToBeAdded);
    }

    public void addLossLayer(final LossLayer lossLayerToBeAdded) {
        if (this.layers.isEmpty()) {
            throw new IllegalArgumentException("Loss layer cannot be the first layer in neural network");
        }

        final LayerBase lastLayer = this.layers.getLast();
        if (!lossLayerToBeAdded.isCompatible(lastLayer)) {
            throw new IllegalArgumentException("Loss layer cannot be placed after "  + lastLayer.getClass());
        }

        this.layers.add(lossLayerToBeAdded);
    }

    public void addSpecialLayer(final LayerBase specialLayerToBeAdded) {
        if (specialLayerToBeAdded instanceof final SoftmaxCategoricalCrossEntropyLayer softmaxCCELayer) {
            if (this.layers.isEmpty()) {
                throw new IllegalArgumentException("Softmax categorical cross entropy layer cannot be the first layer in neural network");
            }

            final LayerBase lastLayer = this.layers.getLast();
            if (!softmaxCCELayer.isCompatible(lastLayer)) {
                throw new IllegalArgumentException("Softmax categorical cross entropy layer cannot be placed after " + lastLayer.getClass());
            }

            this.layers.add(specialLayerToBeAdded);
        } else {
            throw new IllegalArgumentException("Unknown special layer type");
        }
    }

    public void setOptimizer(final OptimizerBase optimizer) {
        if (!this.isLastLayerSuitable()) {
            throw new IllegalArgumentException("Optimizer must be set when all layers are already present");
        } else if (this.optimizer.isPresent()) {
            throw new IllegalArgumentException("Optimizer is already set");
        }

        this.optimizer = Optional.of(optimizer);
    }

    private boolean isLastLayerSuitable() {
        if (this.layers.isEmpty()) {
            return false;
        }

        final LayerBase lastLayer = this.layers.getLast();

        if (lastLayer instanceof LossLayer) {
            return true;
        } else if (lastLayer instanceof SoftmaxCategoricalCrossEntropyLayer) {
            return true;
        }

        return false;
    }

    private static <T extends LayerBase> T getLayerAsType(final List<LayerBase> layers, final int index, final Class<T> type) {
        final LayerBase layer = layers.get(index);

        if (type.isInstance(layer)) {
            return type.cast(layer);
        } else {
            throw new IllegalArgumentException("Layer at index [" + index + "] is not of type [" + type.getName() + "]");
        }
    }

    private static List<Batch> prepareSteps(final Batch batch, final int stepRowsSize) {
        if (batch.getRowsSize() <= stepRowsSize) {
            // Single batch consisting of all rows
            final List<Batch> batchSteps = new ArrayList<>();
            batchSteps.add(batch);
            return batchSteps;
        }

        final int batchRowsSize = batch.getRowsSize();
        final int stepsSize = batchRowsSize / stepRowsSize;

        final List<Batch> batchSteps = new ArrayList<>();
        for (int i = 0; i < stepsSize; ++i) {
            batchSteps.add(new Batch());
        }

        for (int rowIndex = 0; rowIndex < batchRowsSize; ++rowIndex) {
            final int batchStepIndex = rowIndex / stepRowsSize;

            final Batch batchStep = batchSteps.get(Math.min(batchStepIndex, stepsSize - 1));
            batchStep.addRow(batch.getRow(rowIndex));
        }

        return batchSteps;
    }
}
