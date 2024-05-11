import argparse
import json
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Union

import PIL
import torch
import yaml

def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def save_command(path: Union[str, Path], sysargv: List):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == "":
        path.mkdir(exist_ok=True, parents=True)
        path = path / "cmd.txt"
    elif path.suffix == ".txt":
        dirpath = path.parent
        dirpath.mkdir(exist_ok=True, parents=True)
    else:
        raise ValueError

    with open(path, "w") as f:
        cmd = ""
        for arv in sysargv:
            cmd += f"{arv} "
        f.write(f"{cmd}")

    print(f"[*] Saved command at {path}")


def save_config(path: Union[str, Path], dic):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == "":
        path.mkdir(exist_ok=True, parents=True)
        path = path / "config.yaml"
    elif path.suffix == ".yaml" or path.suffix == ".yml":
        dirpath = path.parent
        dirpath.mkdir(exist_ok=True, parents=True)
    else:
        raise ValueError

    with open(path, "w") as f:
        yaml.dump(dic, f)
