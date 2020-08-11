"""paths: Find files in default paths"""
import os

from chia import instrumentation


def maybe_expand_path(path, observable: instrumentation.Observable):
    """Replaces the argument path with a default one if necessary and possible.

    For example, if the user specifies inat_features.h5 and it is not in the CWD,
    we look it up in ~/.chia/ and return this path if the file exists there."""
    observable.log_debug(f"Expanding path for: {path}")

    # (1) Check if path is valid already
    if os.path.exists(path):
        observable.log_debug("  found in CWD.")
        return path

    # (2) Check if path is absolute (even if invalid)
    if os.path.isabs(path):
        observable.log_debug("  is absolute and will not be processed further.")
        return path

    # (3) Check if path is valid in ~/.chia
    chia_directory = os.path.expanduser("~/.chia")
    joined_path = os.path.join(chia_directory, path)
    if os.path.exists(joined_path):
        observable.log_debug(f"  found as {joined_path} in user directory")
        return joined_path

    observable.log_debug("  not found, not processing.")
    return path
