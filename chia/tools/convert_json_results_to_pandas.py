import argparse
import gzip
import json
import typing

import pandas as pd


def process_json(obj: dict) -> typing.List[dict]:
    assert isinstance(obj, dict), "Only dictionary types are supported"

    # Get config dict
    config_dict: dict = dict()
    for config_obj in obj["ConfigMessage"]:
        value = config_obj["value"]
        if not (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, str)
            or isinstance(value, bool)
            or value is None
        ):
            continue
        config_dict[config_obj["field"]] = value

    assert config_dict["RunnerFactory.name"] == "epoch"

    # Check shutdown message
    assert obj["ShutdownMessage"][0]["successful"]

    # Get metadata
    metadata_dict: dict = dict()
    for key in obj["metadata"].keys():
        value = obj["metadata"][key]
        if (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, str)
            or isinstance(value, bool)
            or value is None
        ):
            metadata_dict[key] = value

    ret_val = list()

    for result_obj in obj["ResultMessage"]:
        single_result_dict: dict = dict()
        for key in result_obj["result_dict"].keys():
            value = result_obj["result_dict"][key]
            if (
                isinstance(value, int)
                or isinstance(value, float)
                or isinstance(value, str)
                or isinstance(value, bool)
                or value is None
            ):
                single_result_dict[key] = value
            elif isinstance(value, list):
                for i, element in enumerate(value):
                    single_result_dict[f"{key}_{i}"] = element
            else:
                continue

        single_result_dict["step"] = result_obj["step"]
        single_result_dict["sender"] = result_obj["sender"]
        single_result_dict["timestamp"] = result_obj["timestamp"]

        combined_dict = dict()
        combined_dict.update({f"conf_{k}": v for k, v in config_dict.items()})
        combined_dict.update({f"sres_{k}": v for k, v in single_result_dict.items()})
        combined_dict.update({f"meta_{k}": v for k, v in metadata_dict.items()})
        ret_val += [combined_dict]

    return ret_val


def process_json_path(path: str) -> typing.List[dict]:
    if path.endswith(".json"):
        with open(path) as file_obj:
            return process_json(json.load(file_obj))
    elif path.endswith(".json.gz"):
        with gzip.open(path, "rt") as file_obj:
            return process_json(json.load(file_obj))
    else:
        raise ValueError(
            f"Filename ending of {path} is not supported (only .json and .json.gz are)."
        )


def main(json_paths: typing.List[str], output_path: str):
    data = []
    for path in json_paths:
        data += process_json_path(path)

    data_frames = [pd.DataFrame.from_records(datum, index=[-1]) for datum in data]
    big_df = pd.concat(data_frames, ignore_index=True)

    big_df.to_hdf(output_path, key="main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("chia.tools.convert_json_results_to_pandas")
    parser.add_argument("json_path", nargs="+")
    parser.add_argument("--output_path", help="Output file path", required=True)

    args = parser.parse_args()
    main(json_paths=args.json_path, output_path=args.output_path)
