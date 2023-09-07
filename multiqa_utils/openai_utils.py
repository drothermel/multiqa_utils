import logging
import openai
import os
import time
import jsonlines

import utils.run_utils as ru

def setup_apikey(keyfile="/scratch/ddr8143/.openai_secretkey.txt"):
    openai.api_key = open(keyfile).readlines()[0].strip()
    logging.info(">> API key set")


def prompt_openai(
    prompt, engine="text-davinci-003", max_tokens=256, logprobs=1, temperature=0.0
):
    try:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            logprobs=logprobs,
            temperature=temperature,
            stream=False,
            stop=["<|endoftext|>", "\n\n"],
        )
        return response, response["choices"][0]["text"]
    except Exception as e:
        print(e)
        return None, None


def get_prompt_base(prompt_path):
    return "".join(fu.load_file(prompt_path))


def process_with_prompt(
    data_to_process,
    base_prompt_path,
    make_prompt_fxn, # takes (base_prompt, data_elem)
    outfile,
    engine="text-davinci-003",
    progress_increment=10,
    rate_limit=-1,  # prompts/min, neg = None
    id_field='qid', # used for caching results
):
    base_prompt = get_prompt_base(base_prompt_path)

    # Load any existing results
    mode = "w+"
    if os.path.exists(outfile):
        mode = "a+"
        already_processed = set(
            [d[id_field] for d in fu.load_file(outfile)]
        )
        logging.info(f">> Initial data len: {len(data_to_process):,}")
        data_to_process = [
            d for d in data_to_process if d[id_field] not in already_processed
        ]
        logging.info(
            f"  - after loading: {len(already_processed):,} new len: {len(data_to_process):,}"
        )

    time_per_query = 60.0 / rate_limit
    total_start = time.time()
    with jsonlines.Writer(open(outfile, mode=mode), flush=True) as writer:
        for i, qdata in enumerate(data_to_process):
            ru.processed_log(i, len(data_to_process))

            start = time.time()
            prompt = make_prompt_fxn(base_prompt, qdata)
            if prompt is None:
                continue
            res_text = None
            while res_text is None:
                _, res_text = prompt_openai(prompt, engine=engine)
                if res_text is None and rate_limit > 0:
                    logging.info(">> Hit rate limit, sleeping")
                    time.sleep(120)  # rate limit hit, give it a few mins
                elif res_text is None and rate_limit <= 0:
                    logging.info(">> ERROR: no result text returned, skipping")
                    break  # no expected rate limit but something went wrong

            if res_text is None:
                continue  # skip this example and try others

            writer.write(
                {"prompt": prompt_path, "res_text": res_text, "id": qdata[id_field]}
            )
            remaining_time = time_per_query - (time.time() - start)
            if remaining_time > 0:
                time.sleep(remaining_time)
