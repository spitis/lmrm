from lmrm import *
import os
import json
import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
from typing import Optional, List
from copy import deepcopy

TEMPLATE = {
    "type": "logit_rating",
    "name": "criteria_template",
    "system_prompt": "You are a helpful assistant, that scores other AI assistants based on a given criteria and the quality of their answers.",
    "logit_template": "Rate the quality of the AI assistant's response(s) in the conversation displayed below according to the following criteria:\n\n{criteria}\n\nYour score should reflect the quality of the AI assistant's response(s) with respect to the specific criteria above, ignoring other aspects of the answer (such as overall quality), and should agree with the score provided by a reasonable human evaluator. Please rate the assistant's response(s) on a scale of 1 to {max_score}, where 1 corresponds to extremely poor (criteria is NOT satisfied) and {max_score} corresponds to excellent (criteria is satisfied). Format your answer as: 'I give the assistant a score of X/{max_score}, because...', where X is your score.\n\n[[CONVERSATION]]\n\n{conversation}",
    "logit_completion_template": "I give the assistant a score of ",
    "argmax_score_template": "Rate the quality of the AI assistant's response(s) in the conversation displayed below according to the following criteria:\n\n{criteria}\n\nYour score should reflect the quality of the AI assistant's response(s) with respect to the specific criteria above, ignoring other aspects of the answer (such as overall quality), and should agree with the score provided by a reasonable human evaluator. Begin your evaluation by providing a short explanation. After providing your explanation, please rate the response with respect to the criteria on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[[CONVERSATION]]\n\n{conversation}",
    "argmax_score_template_no_cot": "Rate the quality of the AI assistant's response(s) in the conversation displayed below according to the following criteria:\n\n{criteria}\n\nYour score should reflect the quality of the AI assistant's response(s) with respect to the specific criteria above, ignoring other aspects of the answer (such as overall quality), and should agree with the score provided by a reasonable human evaluator. IMPORTANT: Begin your response by directly rating the AI assistant (with respect to the criteria) on a scale of 1 to 10, strictly following this format: \"[[rating]]\". For example, if you give the AI assistant a score of 5, your response should start with: \"[[5]]\".\n\n[[CONVERSATION]]\n\n{conversation}"
  }

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
  parser.add_argument('--model_name', type=str, default='openbmb/UltraRM-13b')
  #parser.add_argument('--template', type=str, default='basic_template_singleturn')
  parser.add_argument('--model_type', type=str, default='api')
  parser.add_argument('--max_score', type=int, default=7)
  parser.add_argument('--use_cot', action='store_true')

  parser.add_argument('--dataset', type=str, default='RPR')
  parser.add_argument('--split', type=str, default='all')
  parser.add_argument('--max_samples', type=int, default=10)
  parser.add_argument('--tag', type=str, default='')

  parser.add_argument('--no_criteria', action='store_true') # if we want to keep the RM blind to the criteria


  args = parser.parse_args()
  assert args.dataset.lower() == 'rpr'
  #mtbench = MTBench(split=args.split, max_samples=args.max_samples, flatten=True)
  dataset = RPR(split=args.split, max_samples=args.max_samples, flatten=True)

  flattened = False
  if 'deberta' in args.model_name:
    rm = AutoModelForSequenceClassification.from_pretrained(args.model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = RPR(split=args.split, max_samples=args.max_samples, flatten=False)
  elif 'ultra' in args.model_name.lower():
    tokenizer = LlamaTokenizer.from_pretrained("openbmb/UltraRM-13b")
    rm = LlamaRewardModel.from_pretrained("openbmb/UltraRM-13b", load_in_8bit=True)
    dataset = RPR(split=args.split, max_samples=args.max_samples, flatten=False)
  else:
    flattened = True
    if not args.no_criteria:
      rm = LMRM(args.model_name, template=TEMPLATE, model_type=args.model_type, max_score=args.max_score, use_cot=args.use_cot)
    else:
      rm = LMRM(args.model_name, template='basic_template', model_type=args.model_type, max_score=args.max_score, use_cot=args.use_cot)
    dataset = RPR(split=args.split, max_samples=args.max_samples, flatten=True)

    
  print(f"Running {len(dataset)} samples for dataset {args.dataset} split {args.split}...")

  save_file = f"results/{args.dataset}__{'criteria' if not args.no_criteria else 'no_criteria'}__{args.split}__{len(dataset)}__{args.model_name.split('/')[-1]}__{args.model_type}__{args.max_score}__{args.use_cot}__{args.tag}.json"

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
      
      if not flattened:
        assert len(dataset[i]['a']) % 2 == 0

      if not type(dataset[i]['labels']) == list:
        dataset[i]['labels'] = [dataset[i]['labels']]

      if hasattr(rm, 'score'):
        if not args.no_criteria:
          rm.template = deepcopy(TEMPLATE)
          rm.template['logit_template'] = rm.template['logit_template'].replace('{criteria}', dataset[i]['criteria'])
          rm.template['argmax_score_template'] = rm.template['argmax_score_template'].replace('{criteria}', dataset[i]['criteria'])
          rm.template['argmax_score_template_no_cot'] = rm.template['argmax_score_template_no_cot'].replace('{criteria}', dataset[i]['criteria'])
        
        try:
          scores = rm.score([
            dataset[i]['a'],
            dataset[i]['b']
          ])
        except Exception as e:
          raise

        labels = dataset[i]['labels']
      
      else:
        scores = []
        for turn in range(len(dataset[i]['a']) // 2):
          _scores = []
          for letter in ['a', 'b']:
            try:
              prompt_a = flatten_conversation(dataset[i][letter][0:turn*2+1])
              compln_a = flatten_conversation(dataset[i][letter][turn*2+1:turn*2+2])

              if 'deberta' in args.model_name:
                if not args.no_criteria:
                  prompt_a = f"User: Please respond to the following query in order to maximize the score that a reasonable human evaluator would assign given the following criteria: {dataset[i]['criteria']}.\n\nQuery: {prompt_a}"
                else:
                  prompt_a = f"Please continue the following conversation by providing the Assistant's response:\n\n[[Conversation]]\n\n{prompt_a}"
                inputs = tokenizer(prompt_a, compln_a, return_tensors='pt').to('cuda')
                _scores.append(rm(**inputs).logits[0].cpu().item())
              elif 'ultra' in args.model_name.lower():
                if not args.no_criteria:  
                  prompt_a = prompt_a.replace('User:', f"Human: Please respond to the following query in order to maximize the score that a reasonable human evaluator would assign given the following criteria: {dataset[i]['criteria']}.\n\nQuery: ") + f'\n\n{compln_a}'
                else:             
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
      if isinstance(labels, torch.Tensor) or isinstance(labels, np.ndarray):
        labels = labels.tolist()
      if isinstance(scores, torch.Tensor) or isinstance(scores, np.ndarray):
        scores = scores.tolist()

      results[i] = {
        'a': dataset[i]['a'] if flattened else flatten_conversation(dataset[i]['a']),
        'b': dataset[i]['b'] if flattened else flatten_conversation(dataset[i]['b']),
        'criteria': dataset[i]['criteria'],
        'labels': labels,
        'scores': scores
      }

      with open(save_file, 'w') as f:
        json.dump(results, f, indent=2)