package GUI;

import NeuralNetwork.BuildingBlocks.DataList;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.util.List;

public class GUI extends JFrame {
    public static class ChartData {
        private final DataList list;
        private final String name;
        private final int startIndex;

        public ChartData(final DataList list, final String name, final int startIndex) {
            this.list = list;
            this.name = name;
            this.startIndex = startIndex;
        }
    }

    private JPanel panel;
    private ChartPanel chartPanel;

    private JFreeChart customChart;
    private XYSeriesCollection data;

    public GUI() {
        setTitle("GUI");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(1550, 750);
        setLocationRelativeTo(null);
        setContentPane(this.panel);
        setVisible(true);
    }

    public void createUIComponents() {
        final XYSeriesCollection dataset = new XYSeriesCollection();

        this.customChart = ChartFactory.createXYLineChart(
                "Chart",
                "X",
                "Y",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                false,
                false
        );

        this.data = dataset;
        this.chartPanel = new ChartPanel(this.customChart);
    }

    public void show(final List<ChartData> newData) {
        for (final ChartData chartData : newData) {
            final double[] rawValues = chartData.list.getDataListRawValues();
            final XYSeries series = new XYSeries(chartData.name);

            for (int i = 0; i < rawValues.length; ++i) {
                series.add(i + chartData.startIndex, rawValues[i]);
            }

            this.data.addSeries(series);
        }

        this.customChart.fireChartChanged();
    }
}
