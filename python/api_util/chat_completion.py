# Provides terminal-based chat interface for RWKV model.
# Usage: python chat_with_bot.py C:\rwkv.cpp-169M.bin
# Prompts and code adapted from https://github.com/BlinkDL/ChatRWKV/blob/9ca4cdba90efaee25cfec21a0bae72cbd48d8acd/chat.py
import os
import argparse
import pathlib
import copy
import json
import time
import sampling
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from tokenizer_util import add_tokenizer_argument, get_tokenizer
from typing import List, Dict, Optional

# ======================================== Script settings ========================================

# Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
PRESENCE_PENALTY: float = 0.2
# Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
FREQUENCY_PENALTY: float = 0.2

END_OF_LINE_TOKEN: int = 187
END_OF_TEXT_TOKEN: int = 0

# =================================================================================================

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
print(f'System info: {library.rwkv_get_system_info_string()}')

print('Loading RWKV model')
model_path = '/Users/johnnywang/Desktop/workspace/prac/share/rwkv/rwkv.cpp-5world-1b5.bin'
# model_path = '/home/rwkv-model-Q5_1.bin'
model = rwkv_cpp_model.RWKVModel(library, model_path)

tokenizer_decode, tokenizer_encode = get_tokenizer("auto", model.n_vocab)

user = 'Q'
bot = 'A'
separator = ':'

# =================================================================================================


# =================================================================================================

def chat_with_bot(**options):
  DEBUG: bool = options['debug'] or False

  MAX_GENERATION_LENGTH: int = options['max_length'] or 500
  # Sampling temperature. It could be a good idea to increase temperature when top_p is low.
  TEMPERATURE: float = options['temperature'] or 0.8
  # For better Q&A accuracy and less diversity, reduce top_p (to 0.5, 0.2, 0.1 etc.)
  TOP_P: float = options['top_p'] or 0.5
  if DEBUG == True:
    print(f'max:{MAX_GENERATION_LENGTH}, temp:{TEMPERATURE}, top_p:{TOP_P}')

  user_input = options['user_input']
  msg: str = user_input.replace('\\n', '\n').strip()
  if DEBUG == True:
    print(f'User input:{msg}')

  # tokens
  processed_tokens: List[int] = []
  logits: Optional[rwkv_cpp_model.NumpyArrayOrPyTorchTensor] = None
  state: Optional[rwkv_cpp_model.NumpyArrayOrPyTorchTensor] = None

  def process_tokens(_tokens: List[int], new_line_logit_bias: float = 0.0) -> None:
    nonlocal processed_tokens, logits, state
    logits, state = model.eval_sequence_in_chunks(_tokens, state, state, logits, use_numpy=True)

    processed_tokens += _tokens

    logits[END_OF_LINE_TOKEN] += new_line_logit_bias

  new = f'{user}{separator} {msg}\n\n{bot}{separator}'
  process_tokens(tokenizer_encode(new), new_line_logit_bias=-999999999)
  thread = 'chat'

  # Print bot response
  if DEBUG == True:
    print(f'> {bot}{separator}', end='')

  start_index: int = len(processed_tokens)
  accumulated_tokens: List[int] = []
  token_counts: Dict[int, int] = {}
  result: str = ''

  for i in range(MAX_GENERATION_LENGTH):
    for n in token_counts:
      logits[n] -= PRESENCE_PENALTY + token_counts[n] * FREQUENCY_PENALTY

    token: int = sampling.sample_logits(logits, TEMPERATURE, TOP_P)

    if token == END_OF_TEXT_TOKEN:
      print()
      break

    if token not in token_counts:
      token_counts[token] = 1
    else:
      token_counts[token] += 1

    process_tokens([token])

    # Avoid UTF-8 display issues
    accumulated_tokens += [token]

    decoded: str = tokenizer_decode(accumulated_tokens)

    if '\uFFFD' not in decoded:
      if DEBUG == True:
        print(decoded, end='', flush=True)
      result += decoded

      accumulated_tokens = []

    if thread == 'chat':
      if '\n\n' in tokenizer_decode(processed_tokens[start_index:]):
        break

    if i == MAX_GENERATION_LENGTH - 1:
      print()

  return result