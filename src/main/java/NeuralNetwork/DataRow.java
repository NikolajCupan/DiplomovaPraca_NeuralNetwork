package NeuralNetwork;

import Utilities.Helper;

public class DataRow {
    private final Double[] dataValues;

    public DataRow(final int size) {
        this.dataValues = new Double[size];
    }

    public DataRow(final Double[] dataValues) {
        this.dataValues = dataValues;
    }

    public int getDataRowSize() {
        return this.dataValues.length;
    }

    public Double[] getDataRowValues() {
        return this.dataValues;
    }

    public Double getValue(final int index) {
        return this.dataValues[index];
    }

    public void setValue(final int index, final Double value) {
        this.dataValues[index] = value;
    }

    @Override
    public String toString() {
        return Helper.stringifyArray(this.dataValues);
    }
}
