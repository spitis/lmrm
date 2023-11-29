from lmrm import *
import os
import json
import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
from typing import Optional, List

class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,                               
                            )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)
        
        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        rewards = torch.gather(rewards, 1, ends)
        
        return rewards
  
"""
Models:
openbmb/UltraRM-13b
OpenAssistant/reward-model-deberta-v3-large-v2
"""
if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='OpenAssistant/reward-model-deberta-v3-large-v2')
  parser.add_argument('--dataset', type=str, default='mt_bench')
  parser.add_argument('--split', type=str, default='all')
  parser.add_argument('--max_samples', type=int, default=10)
  parser.add_argument('--tag', type=str, default='')


  args = parser.parse_args()
  if args.dataset == 'mt_bench':
    dataset = MTBench(split=args.split, max_samples=args.max_samples, flatten=False)
  elif args.dataset == 'chatbot_arena':
    dataset = ChatbotArena(split=args.split, max_samples=args.max_samples, flatten=False)
  elif args.dataset == 'hhrlhf':
    dataset = HHRLHF(split=args.split, max_samples=args.max_samples, flatten=False)
  else:
    raise

  
  print(f"Running {len(dataset)} samples for dataset {args.dataset} split {args.split}...")

  save_file = f"results/{args.dataset}__{args.split}__{len(dataset)}__{args.model_name.split('/')[-1]}__{args.tag}.json"

  
  if 'deberta' in args.model_name:
    rm = AutoModelForSequenceClassification.from_pretrained(args.model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  elif 'ultra' in args.model_name.lower():
    tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
    rm = LlamaRewardModel.from_pretrained("openbmb/UltraRM-13b", load_in_8bit=True)
  else:
    raise

  if os.path.exists(save_file):
    print(f"loading progress...")
    # load the json file, which contains a single dict
    with open(save_file, 'r') as f:
      results = json.load(f)
  else:
    results = {}

  with torch.no_grad():
    for i in tqdm.tqdm(range(len(dataset))):
      if str(i) in results:
        continue
      
      assert len(dataset[i]['a']) % 2 == 0

      scores = []
      for turn in range(len(dataset[i]['a']) // 2):
        _scores = []
        for letter in ['a', 'b']:
          try:
            prompt_a = flatten_conversation(dataset[i][letter][0:turn*2+1])
            compln_a = flatten_conversation(dataset[i][letter][turn*2+1:turn*2+2])

            if 'deberta' in args.model_name:
              prompt_a = f"Please continue the following conversation by providing the Assistant's response:\n\n[[Conversation]]\n\n{prompt_a}"
              inputs = tokenizer(prompt_a, compln_a, return_tensors='pt').to('cuda')
              _scores.append(rm(**inputs).logits[0].cpu().item())
            elif 'ultra' in args.model_name.lower():
              prompt_a = prompt_a.replace('User:', 'Human:') + f'\n\n{compln_a}'
              inputs = tokenizer(prompt_a, return_tensors='pt').to('cuda')
              _scores.append(rm(**inputs).item())
            else:
              raise

          except Exception as e:
            raise
        scores.append(_scores)
      labels = dataset[i]['labels']
      # convert to list if necessary
      if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

      results[i] = {
        'a': flatten_conversation(dataset[i]['a']),
        'b': flatten_conversation(dataset[i]['b']),
        'labels': labels,
        'scores': scores
      }

      with open(save_file, 'w') as f:
        json.dump(results, f, indent=2)