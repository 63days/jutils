import PIL
from omegaconf import DictConfig, OmegaConf
import rich.syntax
import rich.tree

import wandb


def print_config(
    config,
    fields=(
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


def wandb_log(wandb_on: bool, dic: dict):
    if not wandb_on:
        return

    wandb.log(dic)


def wandb_log_image(wandb_on: bool, log_name: str, images: dict):
    if not wandb_on:
        return

    if isinstance(images, PIL.Image.Image):
        images = [wandb.Image(images)]
    elif isinstance(images, wandb.Image):
        images = [images]
    elif isinstance(images, list):
        if isinstance(images[0], PIL.Image.Image):
            images = [wandb.Image(x) for x in images]

    wandb_log(wandb_on, {log_name: images})


def get_wandb_logger(pl_module):
    for logger in pl_module.logger:
        if isinstance(logger, pl.loggers.wandb.WandbLogger):
            return logger
    return None



