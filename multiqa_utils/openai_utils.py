import openai

def setup_apikey(file="/scratch/ddr8143/.openai_secretkey.txt"):
    openai.api_key = open("/scratch/ddr8143/.openai_secretkey.txt").readlines()[0].strip()
    print(">> API key set")
    

def prompt_openai(prompt, engine='text-davinci-003', max_tokens=256, logprobs=1, temperature=0.0):
    response = None
    try:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            logprobs=logprobs,
            temperature=temperature,
            stream=False,
            stop=["<|endoftext|>", "\n\n"]
        )
    except Exception as e:
        print(e)
    return response, response['choices'][0]['text']