package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

public class Batch {
    private final List<DataRow> batch;

    public Batch() {
        this.batch = new ArrayList<>();
    }

    public void addDataRow(final DataRow dataRow) {
        if (!this.batch.isEmpty()) {
            final int dataRowSize = this.batch.getLast().getDataRowSize();

            if (dataRow.getDataRowSize() != dataRowSize) {
                throw new IllegalArgumentException(
                        "New data row size [" + dataRow.getDataRowSize() + "] is not equal to current data row size [" + dataRowSize + "]"
                );
            }
        }

        this.batch.add(dataRow);
    }

    public DataRow getDataRow(final int index) {
        return this.batch.get(index);
    }

    // Number of data rows currently in batch
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
