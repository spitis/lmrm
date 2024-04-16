python finetune.py --dataset mixed --resume_from_checkpoint True


python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode empty_context --local_folder RM-Mistral-7B_peft_mixed__60000_1e-05_/checkpoint-9000/ --oracle_context --tag mixed9k_oracle



python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --local_folder RM-Mistral-7B_peft_mixed__60000_1e-05_/checkpoint-9000/  --tag mixed9k_ensemble2






python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --local_folder RM-Mistral-7B_peft_mixed__60000_1e-05_/checkpoint-9000/  --tag mixed9k_ensemble
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode empty_context --local_folder RM-Mistral-7B_peft_mixed__60000_1e-05_/checkpoint-9000/  --tag mixed9k_empty
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --local_folder RM-Mistral-7B_peft_mixed__60000_1e-05_/checkpoint-9000/  --tag mixed9k_context
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rpr_criteria --local_folder RM-Mistral-7B_peft_mixed__60000_1e-05_/checkpoint-9000/  --tag mixed9k_rprc
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rpr_scenarios --local_folder RM-Mistral-7B_peft_mixed__60000_1e-05_/checkpoint-9000/  --tag mixed9k_rprs








python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --tag test_baseline --local_folder RM-Mistral-7B_peft_rpr_criteria__30000_2e-05_full/checkpoint-30000/ --max_samples 50
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --tag test_ensemble --ensemble_context --local_folder RM-Mistral-7B_peft_rpr_criteria__30000_2e-05_full/checkpoint-30000/ --max_samples 50
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --tag test_baseline_s --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-15000/ --max_samples 50
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --tag test_ensemble_s --ensemble_context --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-15000/ --max_samples 50



python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --tag ensemble2
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-5000/  --tag ensemble2_fts_5k
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --local_folder RM-Mistral-7B_peft_rpr_criteria__30000_2e-05_full/checkpoint-5000/ --tag ensemble2_ftc_5k
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-15000/ --tag ensemble2_fts_15k
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --local_folder RM-Mistral-7B_peft_rpr_criteria__30000_2e-05_full/checkpoint-30000/ --tag ensemble2_ftc_30k





python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --tag ensemble
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-5000/  --tag ensemble_fts_5k
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --local_folder RM-Mistral-7B_peft_rpr_criteria__30000_2e-05_full/checkpoint-5000/ --tag ensemble_ftc_5k
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-15000/ --tag ensemble_fts_15k
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --local_folder RM-Mistral-7B_peft_rpr_criteria__30000_2e-05_full/checkpoint-30000/ --tag ensemble_ftc_30k



python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode empty_context --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-15000/ --oracle_context --tag finetuned2_oracle_full
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode empty_context --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-15000/ --negative_context --tag finetuned2_negative_full




python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --ensemble_context --tag ensemble





python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode empty_context --oracle_context --tag base_oracle
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode empty_context --negative_context --tag base_negative
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode empty_context --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-2000/ --oracle_context --tag finetuned2_oracle
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode empty_context --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-2000/ --negative_context --tag finetuned2_negative




python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rpr_criteria --context_mode no_context
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rpr_criteria --context_mode context
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rpr_criteria --context_mode empty_context
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rpr_criteria --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-2000/ --context_mode no_context --tag finetuned2
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rpr_criteria --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-2000/ --context_mode context --tag finetuned2
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rpr_criteria --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-2000/ --context_mode empty_context --tag finetuned2



python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode no_context
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode context
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --context_mode empty_context
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-2000/ --context_mode no_context --tag finetuned2
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-2000/ --context_mode context --tag finetuned2
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rewardbench --local_folder RM-Mistral-7B_peft_rpr_scenarios__30000_2e-05_full/checkpoint-2000/ --context_mode empty_context --tag finetuned2

python finetune.py --dataset spitis/rpr_scenarios --tag full
python finetune.py --dataset spitis/rpr_criteria --tag full


python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset mt_bench --local_folder RM-Mistral-7B_peft_rpr_criteria__30000_5e-05/checkpoint-7000/ --tag ftc
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset mt_bench --tag noft # Context ommitted
python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset mt_bench --tag noft --context_mode no_context

python finetune.py --dataset spitis/rpr_scenarios --tag full

# Eval Results
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset rpr_scenarios__context --tag Mar1
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset rpr_scenarios__no_context --tag Mar1
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset rpr_criteria__context --tag Mar1
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset rpr_criteria__no_context --tag Mar1
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset mt_bench__context --tag Mar1
PYTHONPATH=./ python scripts/process_results.py --max_samples 300 --dataset mt_bench__no_context --tag Mar1

# With context, no CoT
PYTHONPATH=./ python scripts/eval_RPR.py --model_name weqweasdas/RM-Mistral-7B --model_type local --dataset rpr_criteria

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