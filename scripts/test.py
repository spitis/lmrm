from lmrm import *


print("Testing LMRM")

dataset = MTBench(split='val', max_samples=100000, flatten=True)

i = np.random.randint(len(dataset))

print(dataset[i]['a'], end='\n\n')
print(dataset[i]['b'], end='\n\n')
print(dataset[i]['labels'], end='\n\n')
      
# try:
#   rm = LMRM('gpt-3.5-turbo', template='basic_template', model_type='openai')
#   print(rm.score(dataset[i]['a']))

#   print('Batched query...\n',
#     rm.score([
#       dataset[i]['a'],
#       dataset[i]['b']
#     ])
#   )
# except:
#   rm = LMRM('gpt-35-turbo', template='basic_template', model_type='openai')
#   print(rm.score(dataset[i]['a']))

#   print('Batched query...\n',
#     rm.score([
#       dataset[i]['a'],
#       dataset[i]['b']
#     ])
#   )


# rm = LMRM('gpt-3.5-turbo', template='basic_template', use_cot=False)
# print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

# rm = LMRM('gpt-3.5-turbo', template='instruction_following')
# print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

# rm = LMRM('gpt-3.5-turbo', template='alpaca_eval')
# print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

# rm = LMRM('gpt-3.5-turbo', template='llm_as_a_judge')
# print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

# rm = LMRM('gpt-4-32k', model_type='openai')
# print(rm.score(dataset[i]['a']))

# print('Batched query...\n',
#   rm.score([
#     dataset[i]['a'],
#     dataset[i]['b']
#   ])
# )

rm = LMRM('meta-llama/Llama-2-70b-chat-hf')
print(rm.score(dataset[i]['a']))

print('Batched query...\n',
  rm.score([
    dataset[i]['a'],
    dataset[i]['b']
  ])
)

rm = LMRM('llama2-7b-chat', model_type='local')
print(rm.score(dataset[i]['a']))

print('Batched query...\n',
  rm.score([
    dataset[i]['a'],
    dataset[i]['b']
  ])
)

# rm = LMRM('meta-llama/Llama-2-13b-chat-hf')
# print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

# rm = LMRM('meta-llama/Llama-2-7b-chat-hf')
# print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))


# rm = LMRM("EleutherAI/pythia-1b", model_type='local')
# print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))

# print('Batched query...\n',
#   rm.score([
#     'User: Hello?\nAssistant: Hi, how may I help you?',
#     'User: How are you?\nAssistant: I was fine until you asked. You stink.'
#   ])
# )