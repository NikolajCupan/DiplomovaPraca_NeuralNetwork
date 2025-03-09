package GUI;

import NeuralNetwork.BuildingBlocks.DataList;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.AbstractRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.util.List;

public class GUI extends JFrame {
    public static class ChartData {
        private final DataList list;
        private final String name;
        private final String color;
        private final int startIndex;

        public ChartData(final DataList list, final String name, final String color, final int startIndex) {
            this.list = list;
            this.name = name;
            this.color = color;
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
        final XYPlot plot = (XYPlot)this.customChart.getPlot();
        final AbstractRenderer renderer = (AbstractRenderer)plot.getRenderer(0);

        for (int i = 0; i < newData.size(); ++i) {
            final ChartData chartData = newData.get(i);

            final double[] rawValues = chartData.list.getDataListRawValues();
            final XYSeries series = new XYSeries(chartData.name);

            for (int j = 0; j < rawValues.length; ++j) {
                series.add(j + chartData.startIndex, rawValues[j]);
            }

            renderer.setSeriesPaint(i, GUI.getColor(chartData.color));

            this.data.addSeries(series);
        }

        this.customChart.fireChartChanged();
    }

    private static Color getColor(final String value) {
        return switch (value) {
            case "blue" -> Color.blue;
            case "red" -> Color.red;
            case "green" -> Color.green;
            case "yellow" -> Color.yellow;
            case "cyan" -> Color.cyan;
            case "orange" -> Color.orange;
            default -> Color.black;
        };
    }
}
