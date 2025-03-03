package NeuralNetwork.Layers.Common;

import NeuralNetwork.BuildingBlocks.*;
import NeuralNetwork.Layers.LayerBase;
import Utilities.CustomMath;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class HiddenLayer extends LayerBase {
    private Optional<RegularizerStruct> regularizerStruct;
    private final List<Neuron> neurons;

    public HiddenLayer() {
        super();

        this.regularizerStruct = Optional.empty();
        this.neurons = new ArrayList<>();
    }

    public HiddenLayer(final int weightsSize, final int neuronsSize) {
        super();

        this.regularizerStruct = Optional.empty();
        this.neurons = new ArrayList<>();

        for (int i = 0; i < neuronsSize; ++i) {
            final Neuron neuron = new Neuron(weightsSize);
            this.neurons.add(neuron);
        }
    }

    public HiddenLayer(final int weightsSize, final int neuronsSize, final long seed) {
        super();

        this.regularizerStruct = Optional.empty();
        this.neurons = new ArrayList<>();

        for (int i = 0; i < neuronsSize; ++i) {
            // Each neuron must get a different seed
            final Neuron neuron = new Neuron(weightsSize, seed + i);
            this.neurons.add(neuron);
        }
    }

    public void addNeuron(final Neuron neuron) {
        if (!this.neurons.isEmpty()) {
            final int weightsSize = this.neurons.getFirst().getWeightsSize();

            if (neuron.getWeightsSize() != weightsSize) {
                throw new IllegalArgumentException("New neuron weights size [" + neuron.getWeightsSize() + "] is not equal to current weights size [" + weightsSize + "]");
            }
        }

        this.neurons.add(neuron);
    }

    public double getRegularizedLoss() {
        if (this.regularizerStruct.isEmpty()) {
            throw new RuntimeException("Cannot calculate regularized loss, regularized in hidden layer is not set");
        }

        double regularizedLoss = 0.0;

        for (final Neuron neuron : neurons) {
            regularizedLoss += this.regularizerStruct.get().getBiasesRegularizerL1() * Math.abs(neuron.getBias());
            regularizedLoss += this.regularizerStruct.get().getBiasesRegularizerL2() * Math.pow(neuron.getBias(), 2.0);

            final DataList weights = neuron.getWeights();
            for (int weightIndex = 0; weightIndex < weights.getDataListSize(); ++weightIndex) {
                regularizedLoss += this.regularizerStruct.get().getWeightsRegularizerL1() * Math.abs(weights.getValue(weightIndex));
                regularizedLoss += this.regularizerStruct.get().getWeightsRegularizerL2() * Math.pow(weights.getValue(weightIndex), 2.0);
            }
        }

        return regularizedLoss;
    }

    public boolean isRegularizerPresent() {
        return this.regularizerStruct.isPresent();
    }

    public List<Neuron> getNeurons() {
        return this.neurons;
    }

    public int getNeuronsSize() {
        return this.neurons.size();
    }

    public int getWeightsSize() {
        if (this.neurons.isEmpty()) {
            throw new IllegalArgumentException("Hidden layer has no neurons, cannot return weights size");
        }

        return this.neurons.getFirst().getWeightsSize();
    }

    public void initializeRegularizer(final RegularizerStruct regularizerStruct) {
        if (this.regularizerStruct.isPresent()) {
            throw new IllegalArgumentException("Hidden layer already has regularizer set");
        }

        this.regularizerStruct = Optional.of(regularizerStruct);
    }

    @Override
    public void forward(final Batch inputBatch) {
        this.setSavedInputBatch(inputBatch);

        final Batch outputBatch = new Batch();

        for (int rowIndex = 0; rowIndex < inputBatch.getRowsSize(); ++rowIndex) {
            final DataList inputRow = inputBatch.getRow(rowIndex);

            final DataList calculatedRow = this.forwardInputRow(inputRow);
            outputBatch.addRow(calculatedRow);
        }

        this.setSavedOutputBatch(outputBatch);
    }

    private DataList forwardInputRow(final DataList inputRow) {
        final DataList outputRow = new DataList(this.neurons.size());

        for (int neuronIndex = 0; neuronIndex < this.neurons.size(); ++neuronIndex) {
            final double neuronOutput = this.neurons.get(neuronIndex).forward(inputRow);
            outputRow.setValue(neuronIndex, neuronOutput);
        }

        return outputRow;
    }

    @Override
    public void backward(final GradientStruct inputGradientStruct) {
        this.setSavedInputGradientStruct(inputGradientStruct);

        final Batch gradientWRTBiases = this.calculateGradientWithRespectToBiases(inputGradientStruct);
        final Batch gradientWRTWeights = this.calculateGradientWithRespectToWeights(inputGradientStruct);
        final Batch gradientWRTInputs = this.calculateGradientWithRespectToInputs(inputGradientStruct);

        if (this.regularizerStruct.isPresent()) {
            this.regularizeGradientWithRespectToBiases(gradientWRTBiases);
            this.regularizedGradientWithRespectToWeights(gradientWRTWeights);
        }

        final GradientStruct outputGradientBatch = new GradientStruct();
        outputGradientBatch.setGradientWithRespectToBiases(gradientWRTBiases);
        outputGradientBatch.setGradientWithRespectToWeights(gradientWRTWeights);
        outputGradientBatch.setGradientWithRespectToInputs(gradientWRTInputs);
        this.setSavedOutputGradientStruct(outputGradientBatch);
    }

    private Batch calculateGradientWithRespectToBiases(final GradientStruct inputGradientStruct) {
        final Batch inputGradientWRTInputs = inputGradientStruct.getGradientWithRespectToInputs();

        final int inputGradientColumnsSize = inputGradientWRTInputs.getColumnsSize();
        final DataList calculatedGradient = new DataList(inputGradientColumnsSize);

        for (int columnIndex = 0; columnIndex < inputGradientColumnsSize; ++columnIndex) {
            final DataList gradientColumn = inputGradientWRTInputs.getColumn(columnIndex);
            calculatedGradient.setValue(
                    columnIndex,
                    CustomMath.sum(gradientColumn)
            );
        }

        final Batch gradientBatch = new Batch();
        gradientBatch.addRow(calculatedGradient);
        return gradientBatch;
    }

    private Batch calculateGradientWithRespectToWeights(final GradientStruct inputGradientStruct) {
        final Batch outputGradientBatch = new Batch();

        final Batch savedInputBatch = this.getSavedInputBatch();
        final int inputBatchColumnsSize = savedInputBatch.getColumnsSize();

        final Batch inputGradientWRTInputs = inputGradientStruct.getGradientWithRespectToInputs();
        final int inputGradientColumnsSize = inputGradientWRTInputs.getColumnsSize();

        for (int inputColumnIndex = 0; inputColumnIndex < inputBatchColumnsSize; ++inputColumnIndex) {
            final DataList inputBatchColumn = savedInputBatch.getColumn(inputColumnIndex);

            final DataList resultRow = new DataList(inputGradientColumnsSize);

            for (int inputGradientIndex = 0; inputGradientIndex < inputGradientColumnsSize; ++inputGradientIndex) {
                final DataList inputGradientColumn = inputGradientWRTInputs.getColumn(inputGradientIndex);

                final double result = CustomMath.dotProduct(inputGradientColumn.getDataListRawValues(), inputBatchColumn.getDataListRawValues());
                resultRow.setValue(
                        inputGradientIndex,
                        result
                );
            }

            outputGradientBatch.addRow(resultRow);
        }

        return outputGradientBatch;
    }

    private Batch calculateGradientWithRespectToInputs(final GradientStruct inputGradientStruct) {
        final Batch outputGradientBatch = new Batch();
        final Batch inputGradientWRTInputs = inputGradientStruct.getGradientWithRespectToInputs();

        for (int rowIndex = 0; rowIndex < inputGradientWRTInputs.getRowsSize(); ++rowIndex) {
            final DataList inputGradientRow = inputGradientWRTInputs.getRow(rowIndex);
            outputGradientBatch.addRow(this.calculateGradientWithRespectToInputs(inputGradientRow));
        }

        return outputGradientBatch;
    }

    private DataList calculateGradientWithRespectToInputs(final DataList inputGradient) {
        final int neuronsSize = this.neurons.size();
        if (neuronsSize != inputGradient.getDataListSize()) {
            throw new IllegalArgumentException("Input gradient size [" + inputGradient.getDataListSize() + "] is not equal to neurons size [" + neuronsSize + "]");
        }

        // Number of connections between one neuron in current layer and all inputs from previous layer
        final int weightsSize = this.neurons.getFirst().getWeightsSize();

        final DataList outputGradient = new DataList(weightsSize);

        for (int columnIndex = 0; columnIndex < weightsSize; ++columnIndex) {
            final DataList columnWeights = this.getColumnWeights(columnIndex);
            outputGradient.setValue(
                    columnIndex,
                    CustomMath.dotProduct(inputGradient.getDataListRawValues(), columnWeights.getDataListRawValues())
            );
        }

        return outputGradient;
    }

    private void regularizeGradientWithRespectToBiases(final Batch gradientWRTBiases) {
        assert(this.regularizerStruct.isPresent());
        final double biasesRegularizerL1 = this.regularizerStruct.get().getBiasesRegularizerL1();
        final double biasesRegularizerL2 = this.regularizerStruct.get().getBiasesRegularizerL2();

        final DataList gradientRow = gradientWRTBiases.getRow(0);

        for (int neuronIndex = 0; neuronIndex < this.neurons.size(); ++neuronIndex) {
            final Neuron neuron = this.neurons.get(neuronIndex);

            if (biasesRegularizerL1 > 0.0) {
                final double originalGradientValue = gradientRow.getValue(neuronIndex);

                if (neuron.getBias() >= 0.0) {
                    gradientRow.setValue(
                            neuronIndex,
                            originalGradientValue + biasesRegularizerL1
                    );
                } else {
                    gradientRow.setValue(
                            neuronIndex,
                            originalGradientValue + biasesRegularizerL1 * -1.0
                    );
                }
            }

            if (biasesRegularizerL2 > 0.0) {
                final double originalGradientValue = gradientRow.getValue(neuronIndex);
                final double updatedGradientValue =
                        originalGradientValue + 2.0 * biasesRegularizerL2 * neuron.getBias();

                gradientRow.setValue(neuronIndex, updatedGradientValue);
            }
        }
    }

    private void regularizedGradientWithRespectToWeights(final Batch gradientWRTWeights) {
        assert(this.regularizerStruct.isPresent());
        final double weightsRegularizerL1 = this.regularizerStruct.get().getWeightsRegularizerL1();
        final double weightsRegularizerL2 = this.regularizerStruct.get().getWeightsRegularizerL2();

        for (int gradientRowIndex = 0; gradientRowIndex < gradientWRTWeights.getRowsSize(); ++gradientRowIndex) {
            final DataList gradientRow = gradientWRTWeights.getRow(gradientRowIndex);
            final DataList columnWeights = this.getColumnWeights(gradientRowIndex);

            if (weightsRegularizerL1 > 0.0) {
                for (int i = 0; i < gradientRow.getDataListSize(); ++i) {
                    final double originalGradientValue = gradientRow.getValue(i);

                    if (columnWeights.getValue(i) >= 0.0) {
                        gradientRow.setValue(
                                i,
                                originalGradientValue + weightsRegularizerL1
                        );
                    } else {
                        gradientRow.setValue(
                                i,
                                originalGradientValue + weightsRegularizerL1 * -1.0
                        );
                    }
                }
            }

            if (weightsRegularizerL2 > 0.0) {
                for (int i = 0; i < gradientRow.getDataListSize(); ++i) {
                    final double originalGradientValue = gradientRow.getValue(i);
                    final double updatedValue =
                            originalGradientValue + 2.0 * weightsRegularizerL2 * columnWeights.getValue(i);

                    gradientRow.setValue(i, updatedValue);
                }
            }
        }
    }

    private DataList getColumnWeights(final int columnIndex) {
        if (this.neurons.isEmpty()) {
            throw new IllegalArgumentException("Neuron list is empty");
        }

        // Number of connections between one input from previous layer and all neurons in current layer
        final int connectionsSize = this.neurons.size();

        final DataList columnWeights = new DataList(connectionsSize);

        for (int rowIndex = 0; rowIndex < connectionsSize; ++rowIndex) {
            final Neuron neuron = this.neurons.get(rowIndex);
            final double weight = neuron.getWeight(columnIndex);

            columnWeights.setValue(rowIndex, weight);
        }

        return columnWeights;
    }

    @Override
    public boolean isCompatible(final LayerBase previousLayer) {
        return previousLayer instanceof ActivationLayer ||
               previousLayer instanceof DropoutLayer;
    }

    @Override
    public String toString() {
        if (this.neurons.isEmpty()) {
            return "{ Layer: empty }";
        }

        final StringBuilder builder = new StringBuilder();

        builder.append("{\n\tHidden layer: input [");
        builder.append(this.neurons.getLast().getWeightsSize());
        builder.append("], output [");
        builder.append(this.neurons.size());
        builder.append("]");

        for (final Neuron neuron : this.neurons) {
            builder.append("\n\t\t").append(neuron);
        }

        builder.append("\n}");

        return builder.toString();
    }
}
