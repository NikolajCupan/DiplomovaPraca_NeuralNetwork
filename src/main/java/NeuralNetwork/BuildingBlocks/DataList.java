package NeuralNetwork.BuildingBlocks;

import Utilities.Helper;

import java.util.Arrays;

public class DataList {
    private final double[] dataValues;

    public DataList(final int size) {
        this.dataValues = new double[size];
    }

    public DataList(final double[] dataValues) {
        this.dataValues = dataValues;
    }

    public void fill(final double value) {
        Arrays.fill(this.dataValues, value);
    }

    public int getDataListSize() {
        return this.dataValues.length;
    }

    public boolean isEmpty() {
        return this.dataValues.length == 0;
    }

    public double[] getDataListRawValues() {
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
