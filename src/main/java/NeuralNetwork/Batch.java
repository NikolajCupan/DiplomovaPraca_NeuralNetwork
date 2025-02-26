package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

public class Batch {
    private final List<DataList> batch;

    public Batch() {
        this.batch = new ArrayList<>();
    }

    public void addRow(final DataList rowToBeAdded) {
        if (!this.batch.isEmpty()) {
            final int rowsSize = this.batch.getFirst().getDataListSize();

            if (rowToBeAdded.getDataListSize() != rowsSize) {
                throw new IllegalArgumentException(
                        "New row size [" + rowToBeAdded.getDataListSize() + "] is not equal to current rows size [" + rowsSize + "]"
                );
            }
        }

        this.batch.add(rowToBeAdded);
    }

    public DataList getRow(final int rowIndex) {
        return this.batch.get(rowIndex);
    }

    public DataList getColumn(final int columnIndex) {
        final DataList outputColumn = new DataList(this.batch.size());

        for (int rowIndex = 0; rowIndex < this.batch.size(); ++rowIndex) {
            outputColumn.setValue(
                    rowIndex,
                    this.batch.get(rowIndex).getValue(columnIndex)
            );
        }

        return outputColumn;
    }

    public int getRowsSize() {
        return this.batch.size();
    }

    public int getColumnsSize() {
        return this.batch.getFirst().getDataListSize();
    }

    @Override
    public String toString() {
        final StringBuilder builder = new StringBuilder();
        builder.append("[\n");

        boolean first = true;
        for (final DataList dataRow : this.batch) {
            if (!first) {
                builder.append("\n");
            }

            builder.append("\t").append(dataRow);
            first = false;
        }

        builder.append("\n]");

        return builder.toString();
    }
}
