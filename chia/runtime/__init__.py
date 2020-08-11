import json
import os.path


def _load_system_config():
    chiarc_filename = os.path.expanduser("~/.chiarc")
    if os.path.exists(chiarc_filename):
        with open(chiarc_filename) as chiarc_file:
            chiarc_content = json.load(chiarc_file)
            assert isinstance(chiarc_content, dict)
            return chiarc_content


__all__ = []
