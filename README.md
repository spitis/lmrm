# language model as a reward model (lmrm)

A simple package to use a language model as a reward model (for language models). 

```
from lmrm import LMRM
rm = LMRM('gpt-4', template='basic_template')
print(rm.score('User: Hello?\nAssistant: Hi, how may I help you?'))
```


### Conversation format

Conversations may be formatted however you like as long as the Assistant's role is clear in the conversation. For example:

```
[USER]
[User Message]

[ASSISTANT]
[Assistant Message]

...
```

They may be several turns long. 