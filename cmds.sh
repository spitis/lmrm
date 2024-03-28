# Eval Results
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset rpr_scenarios__context --tag Mar1
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset rpr_scenarios__no_context --tag Mar1
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset rpr_criteria__context --tag Mar1
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset rpr_criteria__no_context --tag Mar1
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset mt_bench__context --tag Mar1
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset mt_bench__no_context --tag Mar1

# With context, no CoT
PYTHONPATH=./ python scripts/eval_RPR.py --model_name openbmb/UltraRM-13b --model_type local --dataset rpr_scenarios
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-4-turbo --model_type openai --dataset rpr_scenarios #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-35-turbo --model_type openai --dataset rpr_scenarios #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --model_type api --dataset rpr_scenarios #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name meta-llama/Llama-2-70b-chat-hf --model_type api --dataset rpr_scenarios

# Without context, no CoT
PYTHONPATH=./ python scripts/eval_RPR.py --model_name openbmb/UltraRM-13b --model_type local --dataset rpr_scenarios --context_mode no_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-4-turbo --model_type openai --dataset rpr_scenarios --context_mode no_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-35-turbo --model_type openai --dataset rpr_scenarios --context_mode no_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --model_type api --dataset rpr_scenarios --context_mode no_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name meta-llama/Llama-2-70b-chat-hf --model_type api --dataset rpr_scenarios --context_mode no_context

# Empty Context, Sanity Check
PYTHONPATH=./ python scripts/eval_RPR.py --model_name openbmb/UltraRM-13b --model_type local --dataset rpr_scenarios --context_mode empty_context
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-4-turbo --model_type openai --dataset rpr_scenarios --context_mode empty_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-35-turbo --model_type openai --dataset rpr_scenarios --context_mode empty_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --model_type api --dataset rpr_scenarios --context_mode empty_context #

====

# With context, no CoT
PYTHONPATH=./ python scripts/eval_RPR.py --model_name openbmb/UltraRM-13b --model_type local --dataset rpr_criteria #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-4-turbo --model_type openai --dataset rpr_criteria #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-35-turbo --model_type openai --dataset rpr_criteria #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --model_type api --dataset rpr_criteria #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name meta-llama/Llama-2-70b-chat-hf --model_type api --dataset rpr_criteria

# Without context, no CoT
PYTHONPATH=./ python scripts/eval_RPR.py --model_name openbmb/UltraRM-13b --model_type local --dataset rpr_criteria --context_mode no_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-4-turbo --model_type openai --dataset rpr_criteria --context_mode no_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-35-turbo --model_type openai --dataset rpr_criteria --context_mode no_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --model_type api --dataset rpr_criteria --context_mode no_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name meta-llama/Llama-2-70b-chat-hf --model_type api --dataset rpr_criteria --context_mode no_context

====

# With context, no CoT
PYTHONPATH=./ python scripts/eval_RPR.py --model_name openbmb/UltraRM-13b --model_type local --dataset mt_bench #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-4-turbo --model_type openai --dataset mt_bench #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-35-turbo --model_type openai --dataset mt_bench #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --model_type api --dataset mt_bench #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name meta-llama/Llama-2-70b-chat-hf --model_type api --dataset mt_bench

# Without context, no CoT
PYTHONPATH=./ python scripts/eval_RPR.py --model_name openbmb/UltraRM-13b --model_type local --dataset mt_bench --context_mode no_context
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-4-turbo --model_type openai --dataset mt_bench --context_mode no_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-35-turbo --model_type openai --dataset mt_bench --context_mode no_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --model_type api --dataset mt_bench --context_mode no_context
PYTHONPATH=./ python scripts/eval_RPR.py --model_name meta-llama/Llama-2-70b-chat-hf --model_type api --dataset mt_bench --context_mode no_context

# Empty Context, Sanity Check
PYTHONPATH=./ python scripts/eval_RPR.py --model_name openbmb/UltraRM-13b --model_type local --dataset mt_bench --context_mode empty_context
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-4-turbo --model_type openai --dataset mt_bench --context_mode empty_context
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-35-turbo --model_type openai --dataset mt_bench --context_mode empty_context #
PYTHONPATH=./ python scripts/eval_RPR.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --model_type api --dataset mt_bench --context_mode empty_context


====


# With Context, CoT
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-4-turbo --model_type openai --dataset rpr_scenarios --use_cot
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-35-turbo --model_type openai --dataset rpr_scenarios --use_cot
PYTHONPATH=./ python scripts/eval_RPR.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --model_type api --dataset rpr_scenarios --use_cot
PYTHONPATH=./ python scripts/eval_RPR.py --model_name meta-llama/Llama-2-70b-chat-hf --model_type api --dataset rpr_scenarios --use_cot

# Without Context, CoT
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-4-turbo --model_type openai --dataset rpr_scenarios --context_mode no_context --use_cot
PYTHONPATH=./ python scripts/eval_RPR.py --model_name gpt-35-turbo --model_type openai --dataset rpr_scenarios --context_mode no_context --use_cot
PYTHONPATH=./ python scripts/eval_RPR.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --model_type api --dataset rpr_scenarios --context_mode no_context --use_cot
PYTHONPATH=./ python scripts/eval_RPR.py --model_name meta-llama/Llama-2-70b-chat-hf --model_type api --dataset rpr_scenarios --context_mode no_context --use_cot


python scripts/eval_multiturn_lmrm.py --max_samples 2000 --use_cot --model_name gpt-4-32k --model_type openai
python scripts/eval_singleturn_lmrm.py --max_samples 2000 --use_cot --model_name gpt-4-32k --model_type openai --template basic_template_singleturn

# python scripts/eval_multiturn_lmrm.py --max_samples 2000 --model_name mistralai/Mistral-7B-Instruct-v0.1 --model_type local
# python scripts/eval_multiturn_lmrm.py --max_samples 2000 --model_name mistralai/Mistral-7B-Instruct-v0.1 --template goofy_template --model_type local
# python scripts/eval_multiturn_lmrm.py --max_samples 2000 --model_name mistralai/Mistral-7B-Instruct-v0.1 --template adversarial_template --model_type local
# python scripts/eval_multiturn_lmrm.py --max_samples 2000 --model_name mistralai/Mistral-7B-Instruct-v0.1 --template very_adversarial_template --model_type local
# python scripts/eval_multiturn_lmrm.py --max_samples 2000 --model_name mistralai/Mistral-7B-Instruct-v0.1 --template alpaca_eval --model_type local