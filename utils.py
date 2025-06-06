import os
import sys
from openai import OpenAI, RateLimitError

# api_key = os.getenv('OPENAI_API_KEY')
api_key = os.getenv('OPENROUTER_API_KEY')
api_base = os.getenv('OPENROUTER_API_BASE')
# client = OpenAI(api_key=api_key, base_url=api_base)
client = OpenAI(api_key=api_key, base_url=api_base)


from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


Model = Literal["deepseek/deepseek-r1-0528:free", "gpt-4", "gpt-3.5-turbo"]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt, model, seed, temperature=0.0, max_tokens=256, stop_strs=None, is_batched=False, debug=False):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # max_tokens=max_tokens,
            # stop=stop_strs,
            seed=seed,
            temperature=temperature
        )
        if debug:
            print(response.system_fingerprint)
        return response.choices[0].message.content
    except RateLimitError as e:
        print("API 调用已达上限，请等待额度重置后再试")
        raise e

if __name__ == "__main__":
    # print(client.api_key[-4:])

    # response = client.chat.completions.create(
    # model="gpt-3.5-turbo",
    # messages=[
    #     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    #     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
    # ]
    # )

    # print(response.choices[0].message.content)

    response = get_chat("1+1=?", "deepseek/deepseek-r1-0528:free", 4060, debug=True)
    print(response)