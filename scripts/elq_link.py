import logging
import math
import argparse

import utils.file_utils as fu
import multiqa_utils.elq_utils as eu

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logging.info(">> Begin entity linking")

    parser = argparse.ArgumentParser("Entity Link Question Data")
    parser.add_argument(
        "--data_type",
        type=str,
        default="",
        help="(qmp|rqa|qst)_(train|dev)",
    )
    parser.add_argument(
        "--split_num",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--max_num_splits",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--repo_path",
        default="/scratch/ddr8143/repos/BLINK/",
    )

    parser.add_argument("--progress_increment", type=int, default=10)
    args = parser.parse_args()
    path_args = fu.current_default_path_args()

    # Setup and Run Entity Linking
    data_type = args.data_type
    logging.info(f">> Run with data type: {data_type}")

    logging.info(">> Loading question data")
    data_list = fu.get_data(path_args, data_type)
    if args.split_num == -1 or args.max_num_splits == -1:
        logging.info(">> Processing full dataset: {len(data_list):,}")
    else:
        split_size = int(math.ceil(len(data_list) / args.max_num_splits))
        start = split_size * args.split_num
        end = min(split_size * (1 + args.split_num), len(data_list))
        logging.info(f">> Running section {args.split_num} of {args.max_num_splits} of dataset")
        logging.info(f"      which is {split_size:,} elements, from {start:,} to {end:,}")
        data_list = data_list[start:end+1]
    data_in_elq_format = [
        {
            "id": fu.get_data_field(data_type, "id", d),
            "text": fu.get_data_field(data_type, "question", d),
        } for d in data_list
    ]

    elq_kwargs = {
        "eval_batch_size": 1024,
        "split_num": args.split_num,
    }
    num_batches = (len(data_in_elq_format) // elq_kwargs['eval_batch_size']) + 1
    elq_kwargs['num_batches'] = num_batches
    elq_kwargs['print_every'] = max(1, num_batches // 100.0)

    logging.info(">> Running with ELQ Args:")
    elq_args = eu.get_default_args(
        data_type,
        path_args.postp_dir,
        args.repo_path,
        **elq_kwargs,
    )
    for k, v in vars(elq_args).items():
        logging.info(f" - {k+':':20} {v}")
    logging.info('')

    models, tokenizer = eu.load_default_entity_linking_models(
        elq_args
    )
    logging.info(f">> Begin entity linking {len(data_in_elq_format):,} items")
    
    eu.run_elq(elq_args, tokenizer, *models, samples=data_in_elq_format)
    logging.info(f">> Wrote all entity links to: {elq_args.save_preds_dir}")

