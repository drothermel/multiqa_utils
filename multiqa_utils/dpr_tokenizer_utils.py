import json
import dpr
from dpr.options import (
    setup_cfg_gpu,
    get_encoder_params_state_from_cfg,
)
from dpr.models.hf_models import get_bert_tensorizer

from dpr.utils.model_utils import (
    load_states_from_checkpoint,
    get_model_file,
)

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
    model_file = get_model_file(cfg, cfg.checkpoint_file_name)
    saved_state = None
    if model_file:
        logger.info("!! model_file = %s", model_file)
        saved_state = load_states_from_checkpoint(model_file)
        set_cfg_params_from_state(saved_state.encoder_params, cfg)
    return cfg


def initialize_tokenizer(cfg_path=None):
    if cfg_path is None:
        conf = get_tokenizer_config()
    else:
        conf = OmegaConf.load(cfg_path)
    tensorizer = get_bert_tensorizer(conf)
    return tensorizer.tokenizer
