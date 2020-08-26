import argparse
import gzip
import json
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
        row = self.table_metrictypes.currentRow()
        if row >= 0:
            sender = self.table_metrictypes.item(row, 0).text()
            metric = self.table_metrictypes.item(row, 1).text()
            xaxis = "step" if self.radio_metric_step.isChecked() else "timestamp"
            self.plot_metric(sender, metric, xaxis)

    def plot_metric(self, sender, metric, xaxis):
        # Get the data
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
        axes.plot(Xs, Ys)

    def setup_tab_metadata(self):
        self.line_experimentname.setText(self.json_obj["metadata"]["name"])
        self.line_runid.setText(self.json_obj["metadata"]["run_id"])
        self.label_chiaversion.setText(
            f'Using CHIA {self.json_obj["environment"]["chia_version"]}'
        )
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
        self.table_environment.setRowCount(len(environment_dict.keys()))
        for i, (key, value) in enumerate(environment_dict.items()):
            self.table_environment.setItem(i, 0, QtWidgets.QTableWidgetItem(key))
            self.table_environment.setItem(i, 1, QtWidgets.QTableWidgetItem(value))

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
