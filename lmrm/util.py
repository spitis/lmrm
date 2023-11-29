def flatten_conversation(convo):
  s = ''
  for c in convo:
    s += f"{c['role'].capitalize()}: {c['content']}\n\n"
  return s.strip()