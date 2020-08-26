import argparse
import gzip
import json
import typing
from pathlib import Path

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, uic

UI_FILENAME = Path(__file__).parent / "ui" / "json_result_viewer.ui"


class JSONResultViewerUI(QtWidgets.QWidget):
    def __init__(self, json_path):
        super().__init__()
        uic.loadUi(UI_FILENAME, self)

        self.json_path = json_path
        self.json_obj = None
        self.load_json()

    def load_json(self):
        if self.json_path.endswith(".json"):
            with open(self.json_path) as file_obj:
                self.json_obj = json.load(file_obj)
        elif self.json_path.endswith(".json.gz"):
            with gzip.open(self.json_path, "rt") as file_obj:
                self.json_obj = json.load(file_obj)

        # Assertions

        # Raw code
        self.setup_tab_raw_json()

        # Metadata
        self.setup_tab_metadata()

        # Metrics
        self.setup_tab_metrics()

        # Results
        self.setup_tab_result()

    def setup_tab_result(self):
        # Read combinations of metrics and senders
        message_type_dict = dict()
        for message_obj in self.json_obj["ResultMessage"]:
            sender_name = message_obj["sender"]
            for result_key in message_obj["result_dict"].keys():
                key = (sender_name, result_key)
                if key in message_type_dict.keys():
                    message_type_dict[key] += 1
                else:
                    message_type_dict[key] = 1

        self.table_resulttypes.setRowCount(len(message_type_dict.keys()))
        for i, (key, value) in enumerate(message_type_dict.items()):
            self.table_resulttypes.setItem(i, 0, QtWidgets.QTableWidgetItem(key[0]))
            self.table_resulttypes.setItem(i, 1, QtWidgets.QTableWidgetItem(key[1]))
            self.table_resulttypes.setItem(i, 2, QtWidgets.QTableWidgetItem(str(value)))
        self.table_resulttypes.setColumnWidth(0, 150)
        self.table_resulttypes.setColumnWidth(1, 150)
        self.table_resulttypes.setColumnWidth(2, 50)

        self.table_resulttypes.itemSelectionChanged.connect(self.update_results)
        self.radio_result_step.clicked.connect(self.update_results)
        self.radio_result_timestamp.clicked.connect(self.update_results)

    def update_results(self):
        selection_ranges: typing.List[
            QtWidgets.QTableWidgetSelectionRange
        ] = self.table_resulttypes.selectedRanges()

        result_tuples = []
        for selection_range in selection_ranges:
            for row in range(selection_range.topRow(), selection_range.bottomRow() + 1):
                sender = self.table_resulttypes.item(row, 0).text()
                result = self.table_resulttypes.item(row, 1).text()
                xaxis = "step" if self.radio_result_step.isChecked() else "timestamp"

                result_tuples += [(sender, result, xaxis)]

        self.plot_results(result_tuples)
        self.display_raw_results(result_tuples)

    def plot_results(self, result_tuples):
        # Clear the layout
        layout = self.groupbox_result_plot.layout()
        child = layout.takeAt(0)
        if child is not None:
            layout.removeItem(child)
            del child

        # Plot to a new widget
        figure = Figure()
        axes = figure.add_subplot(111)
        layout.addWidget(FigureCanvasQTAgg(figure))

        # Plots
        for result_tuple in result_tuples:
            sender, result, xaxis = result_tuple
            # Get the data
            Xs, Ys = self.extract_results(result, sender, xaxis)

            label = f"{sender}.{result}"
            axes.plot(Xs, Ys, label=label)

        axes.legend()

    def display_raw_results(self, result_tuples):
        if len(result_tuples) != 1:
            self.table_resultraw.clear()
            self.table_resultraw.setColumnCount(0)
            self.table_resultraw.setRowCount(0)
        else:
            result_tuple = result_tuples[0]
            sender, result, xaxis = result_tuple
            # Get the data
            Xs, Ys = self.extract_results(result, sender, xaxis)

            label = f"{sender}.{result}"

            self.table_resultraw.setColumnCount(2)
            self.table_resultraw.setRowCount(len(Xs))
            self.table_resultraw.setHorizontalHeaderItem(
                0, QtWidgets.QTableWidgetItem(xaxis)
            )
            self.table_resultraw.setHorizontalHeaderItem(
                1, QtWidgets.QTableWidgetItem(label)
            )

            for i, (x, y) in enumerate(zip(Xs, Ys)):
                self.table_resultraw.setItem(i, 0, QtWidgets.QTableWidgetItem(str(x)))
                self.table_resultraw.setItem(i, 1, QtWidgets.QTableWidgetItem(str(y)))

    def extract_results(self, result, sender, xaxis):
        valid_messages = [
            message_obj
            for message_obj in self.json_obj["ResultMessage"]
            if message_obj["sender"] == sender
            and result in message_obj["result_dict"].keys()
        ]
        Xs = []
        Ys = []
        for valid_message in valid_messages:
            Ys += [valid_message["result_dict"][result]]
            Xs += [valid_message[xaxis]]
        return Xs, Ys

    def setup_tab_metrics(self):
        # Read combinations of metrics and senders
        message_type_dict = dict()
        for message_obj in self.json_obj["MetricMessage"]:
            key = (message_obj["sender"], message_obj["metric"])
            if key in message_type_dict.keys():
                message_type_dict[key] += 1
            else:
                message_type_dict[key] = 1
        self.table_metrictypes.setRowCount(len(message_type_dict.keys()))
        for i, (key, value) in enumerate(message_type_dict.items()):
            self.table_metrictypes.setItem(i, 0, QtWidgets.QTableWidgetItem(key[0]))
            self.table_metrictypes.setItem(i, 1, QtWidgets.QTableWidgetItem(key[1]))
            self.table_metrictypes.setItem(i, 2, QtWidgets.QTableWidgetItem(str(value)))
        self.table_metrictypes.setColumnWidth(0, 150)
        self.table_metrictypes.setColumnWidth(1, 150)
        self.table_metrictypes.setColumnWidth(2, 50)

        self.table_metrictypes.itemSelectionChanged.connect(self.update_metrics)
        self.radio_metric_step.clicked.connect(self.update_metrics)
        self.radio_metric_timestamp.clicked.connect(self.update_metrics)

    def update_metrics(self):
        selection_ranges: typing.List[
            QtWidgets.QTableWidgetSelectionRange
        ] = self.table_metrictypes.selectedRanges()

        metric_tuples = []
        for selection_range in selection_ranges:
            for row in range(selection_range.topRow(), selection_range.bottomRow() + 1):
                sender = self.table_metrictypes.item(row, 0).text()
                metric = self.table_metrictypes.item(row, 1).text()
                xaxis = "step" if self.radio_metric_step.isChecked() else "timestamp"

                metric_tuples += [(sender, metric, xaxis)]

        self.plot_metrics(metric_tuples)
        self.display_raw_metrics(metric_tuples)

    def plot_metrics(self, metric_tuples):
        # Clear the layout
        layout = self.groupbox_metric_plot.layout()
        child = layout.takeAt(0)
        if child is not None:
            layout.removeItem(child)
            del child

        # Plot to a new widget
        figure = Figure()
        axes = figure.add_subplot(111)
        layout.addWidget(FigureCanvasQTAgg(figure))

        # Plots
        for metric_tuple in metric_tuples:
            sender, metric, xaxis = metric_tuple
            # Get the data
            Xs, Ys = self.extract_metrics(metric, sender, xaxis)

            label = f"{sender}.{metric}"
            axes.plot(Xs, Ys, label=label)

        axes.legend()

    def display_raw_metrics(self, metric_tuples):
        if len(metric_tuples) != 1:
            self.table_metricraw.clear()
            self.table_metricraw.setColumnCount(0)
            self.table_metricraw.setRowCount(0)
        else:
            metric_tuple = metric_tuples[0]
            sender, metric, xaxis = metric_tuple
            # Get the data
            Xs, Ys = self.extract_metrics(metric, sender, xaxis)

            label = f"{sender}.{metric}"

            self.table_metricraw.setColumnCount(2)
            self.table_metricraw.setRowCount(len(Xs))
            self.table_metricraw.setHorizontalHeaderItem(
                0, QtWidgets.QTableWidgetItem(xaxis)
            )
            self.table_metricraw.setHorizontalHeaderItem(
                1, QtWidgets.QTableWidgetItem(label)
            )

            for i, (x, y) in enumerate(zip(Xs, Ys)):
                self.table_metricraw.setItem(i, 0, QtWidgets.QTableWidgetItem(str(x)))
                self.table_metricraw.setItem(i, 1, QtWidgets.QTableWidgetItem(str(y)))

    def extract_metrics(self, metric, sender, xaxis):
        valid_messages = [
            message_obj
            for message_obj in self.json_obj["MetricMessage"]
            if message_obj["sender"] == sender and message_obj["metric"] == metric
        ]
        Xs = []
        Ys = []
        for valid_message in valid_messages:
            Ys += [valid_message["value"]]
            Xs += [valid_message[xaxis]]
        return Xs, Ys

    def setup_tab_metadata(self):
        self.line_experimentname.setText(self.json_obj["metadata"]["name"])
        self.line_runid.setText(self.json_obj["metadata"]["run_id"])
        self.label_chiaversion.setText(
            f'Using CHIA {self.json_obj["environment"]["chia_version"]}'
        )

        # Environment Variables
        environment_dict: dict = dict()
        for key in self.json_obj["environment"]["environment_variables"].keys():
            value = self.json_obj["environment"]["environment_variables"][key]
            if (
                isinstance(value, int)
                or isinstance(value, float)
                or isinstance(value, str)
                or isinstance(value, bool)
                or value is None
            ):
                environment_dict[key] = value
        self.table_environment.setColumnWidth(0, 300)
        self.table_environment.setRowCount(len(environment_dict.keys()))
        for i, (key, value) in enumerate(environment_dict.items()):
            self.table_environment.setItem(i, 0, QtWidgets.QTableWidgetItem(key))
            self.table_environment.setItem(i, 1, QtWidgets.QTableWidgetItem(value))

        # Configuration
        self.table_configuration.setRowCount(len(self.json_obj["ConfigMessage"]))
        self.table_configuration.setColumnWidth(0, 60)
        self.table_configuration.setColumnWidth(1, 300)
        for i, config_message in enumerate(self.json_obj["ConfigMessage"]):
            self.table_configuration.setItem(
                i, 0, QtWidgets.QTableWidgetItem(str(config_message["source"]))
            )
            self.table_configuration.setItem(
                i, 1, QtWidgets.QTableWidgetItem(str(config_message["field"]))
            )
            self.table_configuration.setItem(
                i, 2, QtWidgets.QTableWidgetItem(str(config_message["value"]))
            )

    def setup_tab_raw_json(self):
        reencoded = json.dumps(self.json_obj, indent=2)
        self.raw_json_text.setPlainText(reencoded)


def main(json_path):
    app = QtWidgets.QApplication([])
    window = JSONResultViewerUI(json_path)
    window.show()

    app.exec_()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("chia.tools.json_result_viewer")
    parser.add_argument("json_path", type=str, help="Input file path")
    args = parser.parse_args()
    main(args.json_path)
