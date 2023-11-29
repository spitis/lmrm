from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .util import flatten_conversation

class MultiTurnComparisonDataset(Dataset):
  """
  Subclasses should have a property 'samples' which is a list of:
  {
    'a': str,
    'b': str,
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

