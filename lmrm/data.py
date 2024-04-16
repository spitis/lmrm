from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .util import flatten_conversation
import json

class MultiTurnComparisonDataset(Dataset):
  """
  Subclasses should have a property 'samples' which is a list of:
  {
    'a': str or messages (openai format),
    'b': str or messages (openai format),
    'labels': List[int] if turnwise else int
  }
  """
  
  def __init__(self, split='train', max_samples=None, flatten=False, turnwise=False):
    self.split = split
    self.max_samples = max_samples
    self.flatten = flatten
    self.turnwise = turnwise

  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, idx):
    if self.flatten:
      return {
        'a': flatten_conversation(self.samples[idx]['a']),
        'b': flatten_conversation(self.samples[idx]['b']),
        'labels': torch.tensor(self.samples[idx]['labels'])
      }
    return self.samples[idx]

class MTBench(MultiTurnComparisonDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    dataset = load_dataset("lmsys/mt_bench_human_judgments", split='human')
    qids = np.arange(81, 161)
    np.random.seed(42)
    np.random.shuffle(qids)

    if 'val' in self.split:
      ids = qids[60:]
    elif 'tr' in self.split:
      ids = qids[:60]
    else:
      ids = qids
      
    qs = {}
    for d in dataset:
      i = d['question_id']
      if i in ids:
        key = f"{i}-{d['model_a']}-{d['model_b']}-{d['judge']}"
        if key in qs:
          qs[key].append(d)
        else:
          qs[key] = [d]

    samples = []
    for k, q in qs.items():
      if len(q) != 2:
        continue
      if q[0]['turn'] == 2:
        Q = [q[1], q[0]]
      else:
        Q = q
      
      SCORE_MAP = {
        'model_a': 0,
        'model_b': 1,
        'tie': 2
      }
      samples.append({
        'a': q[0]['conversation_a'],
        'b': q[0]['conversation_b'],
        'labels': [SCORE_MAP[Q[0]['winner']], SCORE_MAP[Q[1]['winner']]]
      })
    self.samples = samples[:self.max_samples] if self.max_samples is not None else samples

class MTBenchCombined(MultiTurnComparisonDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    dataset = load_dataset("lmsys/mt_bench_human_judgments", split='human')
    qids = np.arange(81, 161)
    np.random.seed(42)
    np.random.shuffle(qids)

    if 'val' in self.split:
      ids = qids[60:]
    elif 'tr' in self.split:
      ids = qids[:60]
    else:
      ids = qids
      
    qs = {}
    for d in dataset:
      i = d['question_id']
      if i in ids:
        key = f"{i}-{d['model_a']}-{d['model_b']}"
        if key in qs:
          qs[key].append(d)
        else:
          qs[key] = [d]

    samples = []
    for k, q in qs.items():
      if len(q) != 2:
        continue
      if q[0]['turn'] == 2:
        Q = [q[1], q[0]]
      else:
        Q = q
      
      SCORE_MAP = {
        'model_a': 0,
        'model_b': 1,
        'tie': 2
      }
      samples.append({
        'a': q[0]['conversation_a'],
        'b': q[0]['conversation_b'],
        'labels': [SCORE_MAP[Q[0]['winner']], SCORE_MAP[Q[1]['winner']]]
      })
    self.samples = samples[:self.max_samples] if self.max_samples is not None else samples



class MultiTurnContextComparisonDataset(MultiTurnComparisonDataset):
  """
  Subclasses should have a property 'samples' which is a list of:
  {
    'a': str or messages (openai format),
    'b': str or messages (openai format),
    'context': str,
    'labels': List[int] if turnwise else int
  }
  """
  
  def __getitem__(self, idx):
    if self.flatten:
      return {
        'a': flatten_conversation(self.samples[idx]['a']),
        'b': flatten_conversation(self.samples[idx]['b']),
        'context': self.samples[idx]['context'],
        'labels': torch.tensor(self.samples[idx]['labels']),
        'extra': self.samples[idx].get('extra', None),
      }
    return self.samples[idx]

class RPRCriteria(MultiTurnContextComparisonDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    dataset = load_dataset("spitis/rpr_criteria", split="train")

    keys = list(range(len(dataset)))
    # seed numpy and choose 100 at random
    np.random.seed(42)
    np.random.shuffle(keys)

    samples = []
    num_samples = self.max_samples if self.max_samples is not None else len(keys)

    for k in keys[:num_samples]:
      # randomly sample the criteria
      c = np.random.randint(0, 2)

      samples.append({
        'a': [
          {'role': 'user', 'content': dataset[k]['prompt']},
          {'role': 'assistant', 'content': dataset[k]['response_a']}
        ],
        'b': [
          {'role': 'user', 'content': dataset[k]['prompt']},
          {'role': 'assistant', 'content': dataset[k]['response_b']}
        ],
        'context': dataset[k]['criteria_x'] if c == 0 else dataset[k]['criteria_y'],
        'labels': c
      })
    self.samples = samples

class RPRScenarios(MultiTurnContextComparisonDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    dataset = load_dataset("spitis/rpr_scenarios", split="train")

    keys = list(range(len(dataset)))
    # seed numpy and choose 100 at random
    np.random.seed(42)
    np.random.shuffle(keys)

    samples = []
    num_samples = self.max_samples if self.max_samples is not None else len(keys)

    for k in keys[:num_samples]:
      # randomly sample the criteria
      c = np.random.randint(0, 2)

      samples.append({
        'a': [
          {'role': 'user', 'content': dataset[k]['prompt']},
          {'role': 'assistant', 'content': dataset[k]['more_pref'] if c == 0 else dataset[k]['less_pref']}
        ],
        'b': [
          {'role': 'user', 'content': dataset[k]['prompt']},
          {'role': 'assistant', 'content': dataset[k]['less_pref'] if c == 0 else dataset[k]['more_pref']}
        ],
        'context': dataset[k]['scenario'],
        'labels': c
      })
    self.samples = samples

class RewardBench(MultiTurnContextComparisonDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    dataset = load_dataset("allenai/reward-bench", split='filtered')

    num_samples = self.max_samples if self.max_samples is not None else len(dataset)
    samples = []

    for i, d in enumerate(dataset):
      if i > num_samples:
        break
      samples.append({
        'a': [{'role': 'user', 'content': d['prompt']}, {'role': 'assistant', 'content': d['chosen']}],
        'b': [{'role': 'user', 'content': d['prompt']}, {'role': 'assistant', 'content': d['rejected']}],
        'context': "The response is high quality, relevant, helpful, harmless, detailed, and responsive to the User's request.",
        'labels': 0,
        'extra': {
          'subset': d['subset']
        }
      })
    self.samples = samples

  
class MTBenchTurn2(MultiTurnContextComparisonDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    dataset = load_dataset("lmsys/mt_bench_human_judgments", split='human')
    qids = np.arange(81, 161)
    np.random.seed(42)
    np.random.shuffle(qids)

    if 'val' in self.split:
      ids = qids[60:]
    elif 'tr' in self.split:
      ids = qids[:60]
    else:
      ids = qids
      
    qs = {}
    for d in dataset:
      i = d['question_id']
      if i in ids:
        key = f"{i}-{d['model_a']}-{d['model_b']}-{d['judge']}"
        if key in qs:
          qs[key].append(d)
        else:
          qs[key] = [d]

    samples = []
    for k, q in qs.items():
      if len(q) != 2:
        continue
      if q[0]['turn'] == 2:
        Q = [q[1], q[0]]
      else:
        Q = q
      
      SCORE_MAP = {
        'model_a': 0,
        'model_b': 1,
        'tie': 2
      }
      
      #eliminate ties
      if SCORE_MAP[Q[1]['winner']] == 2:
        continue
      
      samples.append({
        'a': q[0]['conversation_a'][2:],
        'b': q[0]['conversation_b'][2:],
        'context': f"Previously, the user asked: '{q[0]['conversation_a'][0]['content']}', and the assistant gave its response (not shown).",
        'labels': SCORE_MAP[Q[1]['winner']]
      })

    self.samples = samples[:self.max_samples] if self.max_samples is not None else samples
