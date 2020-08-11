import os

import config as pcfg


def get_user_config():
    chiarc_filename = os.path.expanduser("~/.chiarc")
    if os.path.exists(chiarc_filename):
        return pcfg.config_from_json(chiarc_filename, read_from_file=True)
    else:
        return pcfg.config_from_dict(dict())
