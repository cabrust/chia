import argparse
import copy
import json
import os
import os.path
import random
import typing
from functools import reduce


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, typing.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def dict_outer_product(x, y):
    return sum([[update_dict(copy.deepcopy(xel), yel) for yel in y] for xel in x], [])


def fancy_outer_product(levels):
    outer_product_tasks = []
    last_shuffle = False
    for level, do_shuffle in levels:
        if do_shuffle:
            if last_shuffle:
                # Append to last task and reshuffle
                appended_task = dict_outer_product(outer_product_tasks[-1], level)
                random.shuffle(appended_task)
                outer_product_tasks[len(outer_product_tasks) - 1] = appended_task
            else:
                # Add new task and shuffle
                shuffled_level = copy.deepcopy(level)
                random.shuffle(shuffled_level)
                outer_product_tasks += [shuffled_level]
        else:
            # Add new task and don't shuffle
            outer_product_tasks += [level]

        last_shuffle = do_shuffle

    return reduce(dict_outer_product, outer_product_tasks)


def clear_directory(output_dir):
    for the_file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def compare_outer_products(op, op2):
    for cfg in op:
        cfg_str = json.dumps(cfg)
        found = False
        for cfg2 in op2:
            cfg2_str = json.dumps(cfg2)
            if cfg2_str == cfg_str:
                found = True
                break
        if not found:
            raise Exception("OP2 and OP are not equal!")


def compute_grid_levels(grid_spec):
    grid_levels = []
    for key, value in grid_spec.items():
        grid_level = []
        do_shuffle = str(key).endswith("*")
        if str(key).startswith("_"):
            # Fancy outer grid, replace multiple keys
            for inner_value in value:
                grid_level += [inner_value]
        else:
            clean_key = str(key).replace("*", "")
            # Normal inner grid
            for inner_value in value:
                grid_level += [{clean_key: inner_value}]
        grid_levels += [(grid_level, do_shuffle)]
    return grid_levels


def main(grid_spec_path, output_dir, filename_pattern):
    with open(grid_spec_path) as grid_file:
        grid_spec = json.load(grid_file)

    # Compute the different grid levels
    grid_levels = compute_grid_levels(grid_spec)

    # Calculate outer product over grid levels in two ways
    outer_product_with_randomness = fancy_outer_product(grid_levels)
    simple_reduction = reduce(dict_outer_product, [level for level, _ in grid_levels])

    # Sanity check: compare both
    compare_outer_products(outer_product_with_randomness, simple_reduction)

    print("Grid size: %d elements." % len(outer_product_with_randomness))

    # Create location if it doesn't exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Clear output directory
    clear_directory(output_dir)

    for i, element in enumerate(outer_product_with_randomness):
        with open(os.path.join(output_dir, filename_pattern % i), "w") as json_file:
            json.dump(element, json_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("chia.tools.create_config_grid")
    parser.add_argument("--output_dir", default="./configs", help="Output directory")
    parser.add_argument(
        "--filename_pattern",
        default="grid%08d.json",
        help="Pattern for each config file name",
    )
    parser.add_argument("grid_spec", help="Configuration grid specification file")
    args = parser.parse_args()

    main(args.grid_spec, args.output_dir, args.filename_pattern)
