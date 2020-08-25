import argparse
import gzip
import json
from pathlib import Path

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
        reencoded = json.dumps(self.json_obj, indent=2)
        self.raw_json_text.setPlainText(reencoded)

        # Metadata
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
