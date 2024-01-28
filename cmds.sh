python scripts/eval_multiturn_lmrm.py --max_samples 2000 --use_cot --model_name gpt-4-32k --model_type openai
python scripts/eval_singleturn_lmrm.py --max_samples 2000 --use_cot --model_name gpt-4-32k --model_type openai --template basic_template_singleturn

# python scripts/eval_multiturn_lmrm.py --max_samples 2000 --model_name mistralai/Mistral-7B-Instruct-v0.1 --model_type local
# python scripts/eval_multiturn_lmrm.py --max_samples 2000 --model_name mistralai/Mistral-7B-Instruct-v0.1 --template goofy_template --model_type local
# python scripts/eval_multiturn_lmrm.py --max_samples 2000 --model_name mistralai/Mistral-7B-Instruct-v0.1 --template adversarial_template --model_type local
# python scripts/eval_multiturn_lmrm.py --max_samples 2000 --model_name mistralai/Mistral-7B-Instruct-v0.1 --template very_adversarial_template --model_type local
# python scripts/eval_multiturn_lmrm.py --max_samples 2000 --model_name mistralai/Mistral-7B-Instruct-v0.1 --template alpaca_eval --model_type local