package NeuralNetwork;

import Utilities.Helper;

public class DataList {
    private final Double[] dataValues;

    public DataList(final int size) {
        this.dataValues = new Double[size];
    }

    public DataList(final Double[] dataValues) {
        this.dataValues = dataValues;
    }

    public int getDataListSize() {
        return this.dataValues.length;
    }

    public boolean isEmpty() {
        return this.dataValues.length == 0;
    }

    public Double[] getDataListRawValues() {
        return this.dataValues;
    }

    public double getValue(final int index) {
        return this.dataValues[index];
    }

    public void setValue(final int index, final double value) {
        this.dataValues[index] = value;
    }

    @Override
    public String toString() {
        return Helper.stringifyArray(this.dataValues);
    }
}
