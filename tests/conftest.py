"""This file prepares config fixtures for other tests."""

from pathlib import Path

import pytest
import rootutils
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from tests.helpers.run_if import RunIf


@RunIf(min_gpus=1)
@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.data.train_val_test_split = (256, 256, 256)
            cfg.data.batch_size = 2
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
            cfg.model.net.confidence_head_kwargs = {"pairformer_depth": 1}
            cfg.model.net.template_embedder_kwargs = {"pairformer_stack_depth": 1}
            cfg.model.net.msa_module_kwargs = {"depth": 1}
            cfg.model.net.pairformer_stack = {"depth": 2}
            cfg.model.net.diffusion_module_kwargs = {
                "atom_encoder_depth": 1,
                "token_transformer_depth": 1,
                "atom_decoder_depth": 1,
            }
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
            cfg.trainer.devices = 1
            cfg.trainer.precision = 32

            if hasattr(cfg, "callbacks") and hasattr(cfg.callbacks, "learning_rate_monitor"):
                delattr(cfg.callbacks, "learning_rate_monitor")

            if hasattr(cfg, "callbacks") and hasattr(cfg.callbacks, "last_model_checkpoint"):
                cfg.callbacks.last_model_checkpoint.every_n_train_steps = 1

    return cfg


@RunIf(min_gpus=1)
@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

    :return: A DictConfig containing a default Hydra configuration for evaluation.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.data.train_val_test_split = (256, 256, 256)
            cfg.data.batch_size = 2
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
            cfg.model.net.confidence_head_kwargs = {"pairformer_depth": 1}
            cfg.model.net.template_embedder_kwargs = {"pairformer_stack_depth": 1}
            cfg.model.net.msa_module_kwargs = {"depth": 1}
            cfg.model.net.pairformer_stack = {"depth": 2}
            cfg.model.net.diffusion_module_kwargs = {
                "atom_encoder_depth": 1,
                "token_transformer_depth": 1,
                "atom_decoder_depth": 1,
            }
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
            cfg.trainer.devices = 1
            cfg.trainer.precision = 32

            if hasattr(cfg, "callbacks") and hasattr(cfg.callbacks, "learning_rate_monitor"):
                delattr(cfg.callbacks, "learning_rate_monitor")

            if hasattr(cfg, "callbacks") and hasattr(cfg.callbacks, "last_model_checkpoint"):
                cfg.callbacks.last_model_checkpoint.every_n_train_steps = 1

    return cfg


@RunIf(min_gpus=1)
@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:  # type: ignore
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@RunIf(min_gpus=1)
@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:  # type: ignore
    """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
