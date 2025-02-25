package NeuralNetwork.ActivationFunctions;

import NeuralNetwork.DataList;

public class RectifiedLinearUnit implements IActivationFunction {
    @Override
    public DataList apply(final DataList inputList) {
        final DataList outputList = new DataList(inputList.getDataListSize());

        for (int i = 0; i < inputList.getDataListSize(); ++i) {
            outputList.setValue(
                    i,
                    Math.max(0, inputList.getValue(i))
            );
        }

        return outputList;
    }
}
