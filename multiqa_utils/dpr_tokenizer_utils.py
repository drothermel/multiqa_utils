from dpr.options import (
    setup_cfg_gpu,
)
from dpr.models.hf_models import get_bert_tensorizer

from hydra import compose, initialize
from omegaconf import OmegaConf


def get_tokenizer_config():
    # Initialize the tokenizer from hydra
    with initialize(
        version_base="1.1",
        config_path="../../../DPR/conf/",
        job_name="token_ranges",
    ):
        cfg = compose(config_name="extractive_reader_train_cfg", overrides=[])
    cfg = setup_cfg_gpu(cfg)
    return cfg


def initialize_tokenizer(cfg_path=None):
    if cfg_path is None:
        conf = get_tokenizer_config()
    else:
        conf = OmegaConf.load(cfg_path)
    tensorizer = get_bert_tensorizer(conf)
    return tensorizer.tokenizer
