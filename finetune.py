from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os
import random

os.environ["WANDB_PROJECT"] = "rpr"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset as Dset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.utils import PaddingStrategy
from multiprocessing import Pool

from lmrm import *

GENERAL_CONTEXTS = [
  "The response is high quality, relevant, helpful, harmless, detailed, and responsive to the User's request.",
  "The response is helpful and appropriate, as would be expected of a well trained Assistant.",
  "The response is relevant, helpful, and detailed, and is responsive to the User's request.",
  "The User is asking a question to a general purpose Assistant.",
  "The Assistant is a well-trained, high quality model.",
  "The Assistant is a state-of-the-art chatbot.",
  "The Assistant is providing a helpful and harmless response.",
  "The User is asking a question to a general purpose Assistant, and the Assistant is providing a helpful and detailed response.",
  "Exemplifies the Assistant's ability to provide helpful responses with an appropriate level of detail.",
  "Shows the Assistant's ability to provide a helpful response that is relevant to the User's request.",
  "Overall quality",
  "Assistant's overall ability",
  "[Omitted]",
  "[omitted]",
  "No context provided.",
  "N/A"
]

def prompt_with_context(prompt, completion, context=None):
  if context is not None:
    prompt = f"[INST] Please continue the following conversation by providing the Assistant's response, in order to maximize the score that a reasonable human evaluator would assign to the Assistant's response given the following criteria or context: {context}\n\n[[Conversation]]\n\n{prompt} [/INST]" + f' {completion}</s>'
  else:
    prompt = f"[INST] Please continue the following conversation by providing the Assistant's response:\n\n[[Conversation]]\n\n{prompt} [/INST]" + f' {completion}</s>'
  return prompt

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=1e-3)
    dataset: Optional[str] = field(
        default="mixed",
        metadata={"help": "The dataset that you want to train the reward model on."},
    )
    model_name: Optional[str] = field(
        default="weqweasdas/RM-Mistral-7B",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=60000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=1000,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=1600)
    eval_first_step: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to run eval after the first step"},
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    tag: Optional[str] = field(
        default="", metadata={"help": "A tag that will be added to the output directory name."}
    )

PEFT_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense'
    ],
)

class RewardTrainer(Trainer):
  # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
  def compute_loss(self, model, inputs, return_outputs=False):
    rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
    rewards_j, rewards_k = torch.chunk(rewards, 2, dim=0)
    loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
    if return_outputs:
        return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
    return loss

# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_stackexchange(examples):
  new_examples = {
    "input_ids_j": [],
    "attention_mask_j": [],
    "input_ids_k": [],
    "attention_mask_k": [],
  }
  for question, response_j, response_k in zip(examples["question"], examples["response_j"], examples["response_k"]):
    new_examples["prompt_j"] = "Question: " + question + "\n\nAnswer: " + response_j
    new_examples["prompt_k"] = "Question: " + question + "\n\nAnswer: " + response_k

  return new_examples

def preprocess_ctx(example):
  res = {}
  
  prompt, context, compl_j, compl_k = example

  res["prompt_j"] = prompt_with_context(prompt, compl_j, context)
  res["prompt_k"] = prompt_with_context(prompt, compl_k, context)

  return res

   

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
  tokenizer: PreTrainedTokenizerBase
  padding: Union[bool, str, PaddingStrategy] = True
  max_length: Optional[int] = None
  pad_to_multiple_of: Optional[int] = None
  return_tensors: str = "pt"

  def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompts = [feature['prompt_j'] for feature in features] + [feature['prompt_k'] for feature in features]
    batch = self.tokenizer(
        prompts,
        padding=self.padding,
        truncation=True,
        pad_to_multiple_of=self.pad_to_multiple_of,
        return_tensors=self.return_tensors,
    )
    batch["return_loss"] = True
    
    return batch

# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

def process_criteria(dataset):
  examples = []
  for i in range(len(dataset)):
    examples.append((dataset[i]["prompt"], dataset[i]["criteria_x"], dataset[i]["response_a"], dataset[i]["response_b"]))
    examples.append((dataset[i]["prompt"], dataset[i]["criteria_y"], dataset[i]["response_b"], dataset[i]["response_a"]))

  random.seed(42)
  random.shuffle(examples)

  with Pool(num_proc) as p:
    print("Preprocessing data")
    examples = p.map(preprocess_ctx, examples)
    
    print("Creating datasets")
    dataset = Dset.from_list(examples)
  return dataset

def process_scenarios(dataset):
  examples = []
  for i in range(len(dataset)):
    examples.append((dataset[i]["prompt"], dataset[i]["scenario"], dataset[i]["more_pref"], dataset[i]["less_pref"]))

  random.seed(42)
  random.shuffle(examples)

  with Pool(num_proc) as p:
    print("Preprocessing data")
    examples = p.map(preprocess_ctx, examples)
    
    print("Creating datasets")
    dataset = Dset.from_list(examples)
  return dataset

def process_ultrafeedback(dataset):
  examples = []
  # generate a random index from GENERAL_CONTEXTS, one for each datapoint
  c_idx = np.random.choice(len(GENERAL_CONTEXTS), len(dataset))

  for i in range(len(dataset)):
    examples.append((dataset[i]["prompt"], GENERAL_CONTEXTS[c_idx[i]], dataset[i]["chosen"], dataset[i]["rejected"]))

  random.seed(42)
  random.shuffle(examples)

  with Pool(num_proc) as p:
    print("Preprocessing data")
    examples = p.map(preprocess_ctx, examples)
    
    print("Creating datasets")
    dataset = Dset.from_list(examples)
  return dataset

if __name__ == '__main__':
    
  parser = HfArgumentParser(ScriptArguments)
  script_args = parser.parse_args_into_dataclasses()[0]
  set_seed(script_args.seed)
 
  assert script_args.train_subset > 0, "You need to specify a train subset."
  assert script_args.eval_subset > 0, "You need to specify an eval subset."

  num_proc = 6
  if not script_args.dataset == 'mixed':
    dataset = load_dataset(script_args.dataset, split="train")

    if script_args.dataset == "spitis/rpr_criteria":
      dataset = process_criteria(dataset)

    elif script_args.dataset == "spitis/rpr_scenarios":
      dataset = process_scenarios(dataset)

    else:
      raise

  else:
    datasize = script_args.train_subset + script_args.eval_subset
    d1 = load_dataset("spitis/rpr_criteria", split="train").shuffle(seed=42)
    d1 = d1.select(range(0, min(datasize  // 3 , len(d1))))
    d1 = process_criteria(d1)
    d2 = load_dataset("spitis/rpr_scenarios", split="train").shuffle(seed=42)
    d2 = d2.select(range(0, min(datasize  // 3 , len(d2))))
    d2 = process_scenarios(d2)
    d3 = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train").shuffle(seed=42)
    d3 = d3.select(range(0, min(datasize  // 3 , len(d3))))
    d3 = process_ultrafeedback(d3)
    
    print(len(d1), len(d2), len(d3))
    # merge datasets
    dataset =  concatenate_datasets([d1, d2, d3]).shuffle(seed=42)
    del d1, d2, d3

  # Split data
  train_dataset = dataset.select(range(0,len(dataset) - script_args.eval_subset))
  eval_dataset = dataset.select(range(len(dataset) - script_args.eval_subset, len(dataset)))
  del dataset

  # Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
  output_name = (
      f"{script_args.model_name.split('/')[-1]}_peft_{script_args.dataset.split('/')[-1]}__{script_args.train_subset}_{script_args.learning_rate}_{script_args.tag}"
  )

  training_args = TrainingArguments(
      output_dir=output_name,
      learning_rate=script_args.learning_rate,
      per_device_train_batch_size=script_args.per_device_train_batch_size,
      per_device_eval_batch_size=script_args.per_device_eval_batch_size,
      num_train_epochs=script_args.num_train_epochs,
      weight_decay=script_args.weight_decay,
      evaluation_strategy="steps",
      eval_steps=500,
      save_strategy="steps",
      save_steps=1000,
      gradient_accumulation_steps=script_args.gradient_accumulation_steps,
      gradient_checkpointing=script_args.gradient_checkpointing,
      deepspeed=script_args.deepspeed,
      local_rank=script_args.local_rank,
      remove_unused_columns=False,
      label_names=[],
      bf16=script_args.bf16,
      logging_strategy="steps",
      logging_steps=10,
      optim=script_args.optim,
      lr_scheduler_type=script_args.lr_scheduler_type,
      seed=script_args.seed,
      report_to="wandb"
  )

  # Load the value-head model and tokenizer.
  tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

  model = AutoModelForSequenceClassification.from_pretrained(
      script_args.model_name, num_labels=1, quantization_config=BitsAndBytesConfig(load_in_8bit=True)
  )
  model = prepare_model_for_kbit_training(model)
  model = get_peft_model(model, PEFT_CONFIG)
  model.print_trainable_parameters()

  tokenizer.pad_token = tokenizer.eos_token
  model.config.pad_token_id = tokenizer.eos_token_id
  model.config.use_cache = not script_args.gradient_checkpointing

  # Train the model, woohoo.
  trainer = RewardTrainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      compute_metrics=compute_metrics,
      data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
  )

  if script_args.eval_first_step:

      class EvaluateFirstStepCallback(TrainerCallback):
          def on_step_end(self, args, state, control, **kwargs):
              if state.global_step == 1:
                  control.should_evaluate = True

      trainer.add_callback(EvaluateFirstStepCallback())

  trainer.train(script_args.resume_from_checkpoint)

  print("Saving last checkpoint of the model")
  model.save_pretrained(output_name + "_peft_last_checkpoint")