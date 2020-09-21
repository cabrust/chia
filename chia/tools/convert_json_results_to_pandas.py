import argparse
import gzip
import json
import pathlib
import typing

import pandas as pd


def aggregrate_metrics(
    metrics_per_step: typing.Dict[int, dict], max_accumulated_message_count: int
) -> typing.Dict[int, dict]:
    ret_dict = dict()
    accumulator = dict()
    count = dict()

    accumulated_messages = 0

    input_steps = list(sorted(metrics_per_step.keys()))
    for step in input_steps:
        step_input_dict = metrics_per_step[step]
        if step < 0:
            # Don't accumulate step -1, it containes different data
            ret_dict[step] = step_input_dict

        for key, value in step_input_dict.items():
            if key not in accumulator.keys():
                accumulator[key] = 0.0
                count[key] = 0
            accumulator[key] += value
            count[key] += 1

        accumulated_messages += 1

        # Write out aggregate every 100 messages and always for the last step
        if (
            accumulated_messages >= max_accumulated_message_count
            or step == input_steps[-1]
        ):
            step_aggregate_dict = dict()
            for key, acc_value in accumulator.items():
                assert 0 < count[key] <= accumulated_messages
                step_aggregate_dict[key] = acc_value / float(count[key])

            ret_dict[step] = step_aggregate_dict
            accumulated_messages = 0
            count = dict()
            accumulator = dict()

    output_steps = list(sorted(ret_dict.keys()))

    print(f"Accumulated {len(input_steps)} to {len(output_steps)} messages.")
    return ret_dict


def process_json(
    obj: dict, max_accumulated_message_count: int
) -> typing.Tuple[typing.List[dict], typing.List[dict]]:
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

    results_combined_with_metadata = list()

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
        results_combined_with_metadata += [combined_dict]

    metric_per_step = dict()
    for result_obj in obj["MetricMessage"]:
        step = result_obj["step"]
        if step not in metric_per_step.keys():
            metric_per_step[step] = dict()

        metric_per_step[step][result_obj["metric"]] = result_obj["value"]
        metric_per_step[step]["step"] = step

    metric_per_step = aggregrate_metrics(metric_per_step, max_accumulated_message_count)

    metrics_combined_with_metadata = list()
    for step, metric_dict in metric_per_step.items():
        combined_dict = dict()
        combined_dict.update({f"conf_{k}": v for k, v in config_dict.items()})
        combined_dict.update({f"metr_{k}": v for k, v in metric_dict.items()})
        combined_dict.update({f"meta_{k}": v for k, v in metadata_dict.items()})

        metrics_combined_with_metadata += [combined_dict]

    return results_combined_with_metadata, metrics_combined_with_metadata


def process_json_path(
    path: str, max_accumulated_message_count: int
) -> typing.Tuple[typing.List[dict], typing.List[dict]]:
    if path.endswith(".json"):
        with open(path) as file_obj:
            return process_json(json.load(file_obj), max_accumulated_message_count)
    elif path.endswith(".json.gz"):
        with gzip.open(path, "rt") as file_obj:
            return process_json(json.load(file_obj), max_accumulated_message_count)
    else:
        raise ValueError(
            f"Filename ending of {path} is not supported (only .json and .json.gz are)."
        )


def main(
    json_paths: typing.List[str], output_path: str, max_accumulated_message_count: int
):
    # Check if output is available
    if pathlib.Path(output_path).exists():
        raise ValueError(f"File {output_path} already exists! Please delete it first.")

    # Read
    result_data = []
    metric_data = []
    for path in json_paths:
        print(f"Processing {path}...", end="")
        path_results, path_metrics = process_json_path(
            path, max_accumulated_message_count
        )
        result_data += path_results
        metric_data += path_metrics
        print(" done")

    # Concatenate frames
    result_data_frames = [
        pd.DataFrame.from_records(datum, index=[-1]) for datum in result_data
    ]
    result_big_df = pd.concat(result_data_frames, ignore_index=True)

    metric_data_frames = [
        pd.DataFrame.from_records(datum, index=[-1]) for datum in metric_data
    ]
    metric_big_df = pd.concat(metric_data_frames, ignore_index=True)

    # Write to HDF
    result_big_df.to_hdf(output_path, key="main")
    metric_big_df.to_hdf(output_path, key="metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("chia.tools.convert_json_results_to_pandas")
    parser.add_argument("json_path", nargs="+")
    parser.add_argument("--output_path", help="Output file path", required=True)
    parser.add_argument(
        "--accumulate_metric_messages",
        help="Number of accumulated metric messages (default:100)",
        default=100,
    )

    args = parser.parse_args()
    main(
        json_paths=args.json_path,
        output_path=args.output_path,
        max_accumulated_message_count=args.accumulate_metric_messages,
    )
