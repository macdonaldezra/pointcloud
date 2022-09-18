from pointcloud.config import get_model_config


def test_model_config() -> None:
    """
    Test that we can load the default model config.
    """
    config = get_model_config()
    assert config is not None
