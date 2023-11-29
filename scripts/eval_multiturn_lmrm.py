from lmrm import *
import os
import json
import tqdm
import torch

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-70b-chat-hf')
  parser.add_argument('--template', type=str, default='basic_template')
  parser.add_argument('--model_type', type=str, default='api')
  parser.add_argument('--max_score', type=int, default=7)
  parser.add_argument('--use_cot', action='store_true')
  parser.add_argument('--dataset', type=str, default='mt_bench')
  parser.add_argument('--split', type=str, default='all')
  parser.add_argument('--max_samples', type=int, default=10)
  parser.add_argument('--tag', type=str, default='')


  args = parser.parse_args()

  if args.dataset == 'mt_bench':
    dataset = MTBench(split=args.split, max_samples=args.max_samples, flatten=True)
  elif args.dataset == 'chatbot_arena':
    dataset = ChatbotArena(split=args.split, max_samples=args.max_samples, flatten=True)
  elif args.dataset == 'hhrlhf':
    dataset = HHRLHF(split=args.split, max_samples=args.max_samples, flatten=True)
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

    try:
      scores = rm.score([
        dataset[i]['a'],
        dataset[i]['b']
      ])
    except Exception as e:
      raise 

    labels = dataset[i]['labels']
    # convert to list if necessary
    if isinstance(labels, torch.Tensor) or isinstance(labels, np.ndarray):
      labels = labels.tolist()
    if isinstance(scores, torch.Tensor) or isinstance(scores, np.ndarray):
      scores = scores.tolist()
    
    results[i] = {
      'a': dataset[i]['a'],
      'b': dataset[i]['b'],
      'labels': labels,
      'scores': scores
    }

    with open(save_file, 'w') as f:
      json.dump(results, f, indent=2)