from lmrm import *
import os
import json
import tqdm
import torch

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-70b-chat-hf')
  parser.add_argument('--template', type=str, default='basic_template_singleturn')
  parser.add_argument('--model_type', type=str, default='api')
  parser.add_argument('--max_score', type=int, default=7)
  parser.add_argument('--use_cot', action='store_true')
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

  save_file = f"results/{args.dataset}__{args.split}__{len(dataset)}__{args.model_name.split('/')[-1]}__{args.template}__{args.model_type}__{args.max_score}__{args.use_cot}__{args.tag}.json"
  
  rm = LMRM(args.model_name, template=args.template, model_type=args.model_type, max_score=args.max_score, use_cot=args.use_cot)

  if os.path.exists(save_file):
    print(f"loading progress...")
    # load the json file, which contains a single dict
    with open(save_file, 'r') as f:
      results = json.load(f)
  else:
    results = {}

  for i in tqdm.tqdm(range(len(dataset))):
    if str(i) in results:
      continue
      
    assert len(dataset[i]['a']) % 2 == 0

    scores = []
    for turn in range(len(dataset[i]['a']) // 2):
      scores.append(list(rm.score([
        flatten_conversation(dataset[i]['a'][0:turn*2+2]),
        flatten_conversation(dataset[i]['b'][0:turn*2+2])
      ])))

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