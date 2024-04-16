import os, json
import glob
import numpy as np

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='mt_bench')
  parser.add_argument('--model', type=str, default='all')
  parser.add_argument('--split', type=str, default='all')
  parser.add_argument('--max_samples', type=str, default='100')
  parser.add_argument('--tag', type=str, default='')
  parser.add_argument('--dedupe_mtbench', action='store_true')
  parser.add_argument('--ensemble', action='store_true') #for ensemble scores
  args = parser.parse_args()


  files = glob.glob(f"results/{args.dataset}__{args.split}__{args.max_samples}__*__{args.tag}.json")
  print(f"Found {len(files)} files")

  if args.model != 'all':
    files = [f for f in files if args.model in f]

  for filename in files:
    with open(filename, 'r') as f:
      results = json.load(f)

    turnwise_scores = []
    turnwise_labels = []
    all_scores = []
    all_labels = []

    if args.dedupe_mtbench:
      # deduplicate
      counter = {}
      deduped = {}
      for k, v in results.items():
        new_key = hash(v['a'] + v['b'])
        if new_key in deduped:
          # dedupe
          labels = np.array(v['labels']) - 0.5
          labels[labels > 1] = 0
          labels = (labels + np.array(deduped[new_key]['labels'])*counter[new_key]) / (counter[new_key] + 1)
          deduped[new_key]['labels'] = labels.tolist()
          counter[new_key] += 1
        else:
          deduped[new_key] = v
          labels = np.array(v['labels']) - 0.5
          labels[labels > 1] = 0 # ties
          deduped[new_key]['labels'] = labels.tolist()
          counter[new_key] = 1

      results = deduped

    for r in results.values():
      # filter out any pure ties
      scores = r['scores']
      # convert any nan to 5.0 for gpt...
      if 'singleturn__openai' in filename:
        scores = [[5.0, s[1]] if s[0] is None else s for s in scores]
        scores = [[s[0], 5.0] if s[1] is None else s for s in scores]
      elif 'openai' in filename:
        scores = [5.0 if s is None else s for s in scores]
      
      if not type(r['labels']) == list:
        r['labels'] = [r['labels']]
      if not args.dedupe_mtbench:
        labels = r['labels']
        labels = np.array(labels) - 0.5
        labels[labels > 1] = 0 # ties
      else:
        labels = np.array(r['labels'])
      if isinstance(scores[0], list): # turnwise scores
        try:
          all_scores.append(np.array(scores).mean(0)) # avg scores for multiturn eval
        except: 
          print(f"WARNING: could not average scores for {filename}")
        if len(labels) == len(scores):
          # we can compute turnwise accuracy
          turnwise_scores += scores
          turnwise_labels += labels.tolist()
      else:
        all_scores.append(scores)
      if len(labels.shape) == 1:
        all_labels.append(labels.mean())
      else:
        all_labels.append(labels)

    print(f"\n{filename}:")

    if args.ensemble and len(all_scores[0].shape) == 2:
      all_scores = [s.mean(1) for s in all_scores]
      turnwise_scores = []
    
    
    try:
      if len(turnwise_scores):
        not_turnwise_ties = np.abs(turnwise_labels) > 0.1
        turnwise_scores = np.array(turnwise_scores)
        turnwise_scores = (turnwise_scores[:, 1] - turnwise_scores[:, 0])[not_turnwise_ties]
        turnwise_preds = turnwise_scores > 0
        turnwise_labels = (np.array(turnwise_labels) > 0)[not_turnwise_ties]
        turnwise_acc = (turnwise_preds == turnwise_labels).mean()
        print(f"Turnwise accuracy: {turnwise_acc:.4f} (N = {len(turnwise_scores)})")

      not_ties = np.abs(all_labels) > 0.05
      #mild_pref = np.abs(all_labels) > 0.24
      strict_pref = np.abs(all_labels) > 0.49
      scores = np.array(all_scores)
      scores = (scores[:, 1] - scores[:, 0])
      preds = scores > 0
      labels = (np.array(all_labels) > 0)

      acc = (preds[not_ties] == labels[not_ties]).mean()
      #acc_mild = (preds[mild_pref] == labels[mild_pref]).mean()
      acc_strict = (preds[strict_pref] == labels[strict_pref]).mean()

      print(f"Mild pref accuracy: {acc:.4f} (N = {not_ties.sum()})")
      #print(f"Mild pref accuracy: {acc_mild:.3f} (N = {mild_pref.sum()})")
      print(f"Strict pref accuracy: {acc_strict:.4f} (N = {strict_pref.sum()})")
  
    except Exception as e:
      print(f"WARNING: could not compute turnwise accuracy for {filename}")
