//package NeuralNetwork.OldLayers.ActivationLayer;
//
//import NeuralNetwork.ActivationFunctions.IActivationFunction;
//import NeuralNetwork.BuildingBlocks.Batch;
//import NeuralNetwork.BuildingBlocks.DataList;
//import NeuralNetwork.OldLayers.AbstractLayerBase;
//import NeuralNetwork.BuildingBlocks.GradientStruct;
//
//public class ActivationLayer extends AbstractLayerBase {
//    private final IActivationFunction activationFunction;
//
//    public ActivationLayer(final IActivationFunction activationFunction) {
//        super();
//        this.activationFunction = activationFunction;
//    }
//
//    @Override
//    protected DataList forward(final DataList inputRow) {
//        return this.activationFunction.activate(inputRow);
//    }
//
//    @Override
//    public GradientStruct backward(final Batch inputGradientBatch) {
//        return this.activationFunction.backward(inputGradientBatch, this.getSavedInputBatch(), this.getSavedOutputBatch());
//    }
//}
