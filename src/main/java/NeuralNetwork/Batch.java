package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

public class Batch {
    private final List<DataRow> batch;

    public Batch() {
        this.batch = new ArrayList<>();
    }

    public void addInputRow(final DataRow inputRow) {
        if (!this.batch.isEmpty()) {
            final int inputRowSize = this.batch.getLast().getDataRowSize();

            if (inputRow.getDataRowSize() != inputRowSize) {
                throw new IllegalArgumentException(
                        "New input row size [" + inputRow.getDataRowSize() + "] is not equal to current input row size [" + inputRowSize + "]"
                );
            }
        }

        this.batch.add(inputRow);
    }

    public DataRow getInputRow(final int index) {
        return this.batch.get(index);
    }

    // Number of input rows currently in batch
    public int getBatchSize() {
        return this.batch.size();
    }

    @Override
    public String toString() {
        final StringBuilder builder = new StringBuilder();
        builder.append("[\n");

        boolean first = true;
        for (final DataRow dataRow : this.batch) {
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
