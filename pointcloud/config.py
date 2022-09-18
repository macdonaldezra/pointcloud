import os
from pathlib import Path
from typing import IO, Any, Dict, Union

from pydantic import BaseModel, Field
from ruamel.yaml import YAML
from ruamel.yaml.composer import ComposerError
from ruamel.yaml.parser import ParserError
from ruamel.yaml.scanner import ScannerError

from pointcloud.utils.logging import get_logger

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
LOGGER = get_logger()


class TrainingConfig(BaseModel):
    feature_dim: int
    epochs: int
    learning_rate: float
    learning_rate_weight_decay: float
    batch_size: int
    pointcloud_input_size: int
    save_frequency: int

    data_path: Path = Field(..., env="DATA_PATH")
    output_path: Path = Field(..., env="OUTPUT_PATH")


def parse_yaml(data: Union[IO, bytes]) -> Union[None, Dict[str, Any]]:
    """
    Parse bytes or input data that ideally contains valid yaml.
    """
    try:
        yaml = YAML(typ="safe")
        return yaml.load(data)
    except (ScannerError, ParserError) as err:
        LOGGER.error(f"Error while trying to parse YAML:\n {err}")
        return None
    except ComposerError as err:
        LOGGER.error(f"Provided more than one YAML document:\n {err}")
        return None


def read_yaml(filepath: Path) -> Union[None, Dict[str, Any]]:
    """
    Read in a YAML file and return file contents in a dict.
    """
    try:
        fptr = open(filepath, "r")
        data = parse_yaml(fptr)
    except FileNotFoundError as err:
        LOGGER.error(f"File {err.filename} not found.")
        return None
    except IOError as err:
        LOGGER.error(f"Unable to parse contents of {err.filename}.")
        return None

    return data


def get_model_config() -> TrainingConfig:
    """
    Try and parse
    """
    if "MODEL_CONFIG" not in os.environ:
        LOGGER.warning(
            "Variable MODEL_CONFIG not found. Falling back to default config file from project root."
        )

    config_path = Path(
        os.getenv("MODEL_CONFIG", PROJECT_ROOT / "config_files" / "default.yaml")
    )
    try:
        config_data = read_yaml(config_path)
    except OSError:
        LOGGER.error("Unable to parse config from provided filepath.")
        raise ValueError("Unable to load settings.")

    if not config_data:
        LOGGER.error(
            "Returned config is empty. Please check the format of your config file and try again."
        )

    config = TrainingConfig.parse_obj(config_data)

    return config
