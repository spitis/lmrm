import os
import re
import json
import glob
import openai
import numpy as np
from huggingface_hub import InferenceClient
import dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

dotenv.load_dotenv(override=True)
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.organization = os.getenv('OPENAI_ORGANIZATION')

LLAMA_TEMPLATE = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST] """

@retry(
    reraise=True,
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=(retry_if_exception_type(openai.error.Timeout)
        | retry_if_exception_type(openai.error.APIError)
        | retry_if_exception_type(openai.error.APIConnectionError)
        | retry_if_exception_type(openai.error.RateLimitError)),

)
def chat_decode(input: list, max_length: int = 128, temp: float = 0, stop: str | list[str] | None = None, n: int = 1, engine='gpt-4'):

    if openai.api_type == 'azure':
      response = openai.ChatCompletion.create(
        engine=engine,
        messages=input,
        max_tokens=max_length,
        temperature=temp,
        stop=stop,
        n=n)
    else:
      response = openai.ChatCompletion.create(
        model=engine,
        messages=input,
        max_tokens=max_length,
        temperature=temp,
        stop=stop,
        n=n)
    
    return [response["choices"][j]["message"]["content"] for j in range(len(response["choices"]))]

def get_template(template_name):
  templates = {}
  dirname = os.path.dirname(__file__)
  for _f in glob.glob(os.path.join(dirname, 'templates', '*.json')):
    with open(_f, 'r') as f:
      _templates = {t['name']: t for t in json.load(f)}
    templates.update(_templates)

  return templates[template_name]
  
class LMRM():
  def __init__(self, model: str, template: str | dict = 'basic_template', use_cot: bool = True):
    self.model = None
    self.openai = False
    self.use_cot = use_cot

    try:
      openai.Model.retrieve(model)
      self.model = model
      self.openai = True
      print(f"Found {model} in OpenAI directory! Treating as OpenAI model.")
    except openai.error.AuthenticationError as e:
      raise e
    except:
      print(f"Treating {model} as a HuggingFace model")
      # it is a huggingface hub model
      self.model = InferenceClient(model=model, token=os.getenv('HUGGINGFACE_TOKEN'))

    if isinstance(template, str):
      self.template = get_template(template)
    else:
      self.template = template

  def score(self, conversation: str):
    if self.openai:
      return self.score_openai(conversation)
    else:
      return self.score_huggingface(conversation)
  
  def score_openai(self, conversation: str):
    user_message = self.template['argmax_score_template'] if self.use_cot else self.template['argmax_score_template_no_cot']
    user_message = user_message.format(conversation = conversation)
    messages = [
      {"role": "system", "content": self.template['system_prompt']},
      {"role": "user", "content": user_message}
    ]

    response = chat_decode(messages, engine=self.model)
    res = re.search('\[\[(.*)\]\]', response[0])
    try:
      res = float(res.group(1))
      assert 0 <= res <= 10
      return res
    except Exception as e:
      raise Exception(f"GPT did not return a proper score. Exception: {e}. GPT's full response: {response[0]}")
    
  def score_huggingface(self, conversation: str, temperature=0.5):
    user_message = self.template['logit_template'].format(conversation = conversation)
    prompt = LLAMA_TEMPLATE.format(system_prompt=self.template['system_prompt'], user_message=user_message)
    prompt += self.template['logit_completion_template']
    output = self.model.post(json={'inputs': prompt, 'parameters':{'top_n_tokens': 5, 'details': True, 'max_new_tokens': 1}})
    top_tokens = json.loads(output)[0]['details']['top_tokens'][0]
    top_tokens = {t['text']:t['logprob'] for t in top_tokens}
    scores = np.array([top_tokens.get(str(i), -100.) for i in range(1,6)])
    scores = scores / temperature
    scores -= np.max(scores)
    scores = np.exp(scores)
    return np.arange(5).dot(scores) * 9/4 + 1 # normalize to 1-10


if __name__ == '__main__':
  print("Testing LMRM")
  import time

  rm = LMRM('gpt-3.5-turbo', template='basic_template')
  print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

  rm = LMRM('gpt-3.5-turbo', template='basic_template', use_cot=False)
  print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

  rm = LMRM('gpt-3.5-turbo', template='instruction_following')
  print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

  rm = LMRM('gpt-3.5-turbo', template='alpaca_eval')
  print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

  rm = LMRM('gpt-3.5-turbo', template='llm_as_a_judge')
  print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))
  
  rm = LMRM('gpt-4')
  print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

  rm = LMRM('meta-llama/Llama-2-70b-chat-hf')
  print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

  time.sleep(1)
  rm = LMRM('meta-llama/Llama-2-13b-chat-hf')
  print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

  time.sleep(1)
  rm = LMRM('meta-llama/Llama-2-7b-chat-hf')
  print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))