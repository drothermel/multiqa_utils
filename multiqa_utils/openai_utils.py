import openai
import os
import time
import jsonlines

import multiqa_utils.general_utils as gu


def setup_apikey(keyfile="/scratch/ddr8143/.openai_secretkey.txt"):
    openai.api_key = (
        open(keyfile).readlines()[0].strip()
    )
    print(">> API key set")


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

    
def get_simple_prompt_v1(qdata):
    prompt_str = """Instruction: For the input question where the answer is a list of entities, output the sampled answer wikipedia pages, answer type and wikipedia pages that must be used to find the answers.

Question: Which movie, clip, TV show etc. had Chezhiyan as director of photography?
Sampled Answers: To Let (film), Kalloori, Thenmerku Paruvakaatru, A Little Dream, Paradesi (2013 film)
Answer Type: movie, clip, TV show
Pages: Chezhiyan

Question: Which album has Kodak Black as performer?
Sampled Answers: Lil B.I.G. Pac, Painting Pictures, Bill Israel, Dying to Live (Kodak Black album)
Answer Type: album
Pages: Kodak Black

Question: Which spatial entity is located in Tapoa Province?
Sampled Answers: Kantchari Department, Namounou Department, Tansarga Department, Botou Department, Tambaga Department, Logobou Department, Diapaga Department
Answer Type: department
Pages: Tapoa Province

Question: Which competition is located in Hard Rock Stadium?
Sampled Answers: Super Bowl XLIV, Super Bowl XXIII, 2021 Miami Open, 2019 Miami Open, Floyd Mayweather Jr. vs. Logan Paul, Super Bowl XXXIII
Answer Type: competition
Pages: Hard Rock Stadium

Question: Who received the award Sandover Medal?
Sampled Answers: Haydn Bunton Sr., John Loughridge, Lin Richards, Neville Beard, Shane Beros, George Krepp, Ted Kilmurray, Polly Farmer
Answer Type: person
Pages: Sandover Medal

Question: Who was born in Hillsdale?
Sampled Answers: Charles Grosvenor, Tanner McEvoy, Jesse Van Saun, Jean Carol, Kathleen Noone
Answer Type: person
Pages: Hillsdale, New Jersey

Question: Which groups were formed in Preston?
Sampled Answers: Sharp’s Commercials, Failsafe (UK band), The KBC, Matalan, Xentrix
Answer Type: group
Pages: Preston, Lancashire

Question: Who played the sport kabaddi?
Sampled Answers: Sachin Tanwar, Dharmaraj Cheralathan, Meraj Sheykh, Deepika Henry Joseph, Jasmer Singh, Maleka Parvin, Fatema Akhter Poly, Sharmin Sultana Rima, Harjeet Brar Bajakhana, Dabang Delhi
Answer Type: player
Pages: Kabaddi

Question: Who played in the women’s doubles 2007 European Junior Badminton Championships?
Sampled Answers: Gabriela Stoeva/Stefani Stoeva, Petya Nedelcheva/Anastasia Russkikh, Anastasia Chervyakova/Anastasia Prokopenko, Anastasia Kharitonova/Anastasia Cherniaeva, Anastasia Kharitonova/Anastasia Cherniaeva
Answer Type: player
Pages: 2007 European Junior Badminton Championships

Question: """
    q_text = qdata["question_text"].strip()
    if q_text[-1] != "?":
        q_text += "?"
    return prompt_str + q_text + "\n" + "Sampled Answers:"


def get_pred_type_from_decomp(decomp_lines):
    if "[ANS1]" in decomp_lines[-1]:
        return "composition"
    elif "None" in decomp_lines[-1]:
        return "simple"
    return "intersection"


def process_with_prompt(
    data_to_process, 
    outfile,
    dataset="qampari",
    progress_increment=10,
    engine='text-davinci-003',
    rate_limit=19,
):
    assert dataset == "qampari"
    mode = 'w+'
    
    if os.path.exists(outfile):
        mode = 'a+'
        already_processed = set([d['qid'] for d in gu.loadjsonl(outfile)])
        print("Initial data len:", len(data_to_process))
        data_to_process = [d for d in data_to_process if d['qid'] not in already_processed]
        print("  - after loading:", len(already_processed), "new len:", len(data_to_process))

    
    time_per_query = 60.0 / rate_limit
    total_start = time.time()
    start = time.time()
    with jsonlines.Writer(open(outfile, mode=mode), flush=True) as writer:
        for i, qdata in enumerate(data_to_process):
            if i % progress_increment == 0:
                print(f">> [{(time.time() - total_start)/60.0:0.1f}] elem {i:,} / {len(data_to_process):,}")
            
            qprompt = get_simple_prompt_v1(qdata)
            _, res_text = prompt_openai(qprompt, engine=engine)
            while res_text is None:
                print("  -> Hit strange rate limit, sleep for two minutes and try again.")
                time.sleep(120)
                _, res_text = prompt_openai(qprompt, engine=engine)
                print("        ===> okkk, lets go!")
            
            writer.write({'prompt': qprompt, 'res_text': res_text, 'qid': qdata['qid']})
            end = time.time()
            print("Raw time:", end - start)
            while end - start < time_per_query:
                time.sleep(5.0)
                end = time.time()
            print("  -> Total time:", end-start)
            start = end
    print("Fnished Processing & Wrote:", outfile)
