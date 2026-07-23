import pickle
import sys, os
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from pyctcdecode import build_ctcdecoder
from characterDefinitions import getHandwritingCharacterDefinitions
cache_dir = '/data/hossein/mm_project/cache/'

device = "cuda" if torch.cuda.is_available() else "cpu"
gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl", cache_dir=cache_dir)
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2-xl", cache_dir=cache_dir).to(device)
gpt_model.eval()

def compute_gpt2_score(text):
    if not text.strip():
        return 9999.0
    text_formatted = text[0].upper() + text[1:] if len(text) > 0 else text
    inputs = gpt_tokenizer(text_formatted, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] <= 1:
        return 9999.0
    with torch.no_grad():
        outputs = gpt_model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss.item() * (input_ids.shape[1] - 1)
    return neg_log_likelihood

labels = [
    "",
    " ",
    "'",
    ",",
    ".",
    "?",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z"
]

prefix = ''
charDef = getHandwritingCharacterDefinitions()
rootDir = "/data/hossein/mm_project" + '/handwritingBCIData/'
langModelDir = rootDir + 'BigramLM'
i = 0
folder = f"nlp10-l208-rdpns{i}ro"

with open(f'/data/hossein/mm_project/speech_gru_cebra/{folder}/{prefix}logits', 'rb') as f:
    rnn_outputs = pickle.load(f)

for key in ["logits"]:
    new_list = []
    for item in rnn_outputs[key]:
        new_list.append(np.array(item))
    rnn_outputs[key] = new_list

ACOUSTIC_SCALE = 1.0
BEAM_WIDTH = 65
N_BEST = 128
ALPHA = 1.0 / ACOUSTIC_SCALE
BETA = 0.0

decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path=langModelDir + "/webTextSentences_tokenized-2gram-50000.arpa",
    alpha=ALPHA,
    beta=BETA
)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

n = len(rnn_outputs["logits"])
for i in range(n):
    logits = rnn_outputs["logits"][i]
    logits = logits[:, charDef['idxToKaldi']]
    if not isinstance(logits, np.ndarray):
        logits = logits.numpy()
    probs = softmax(logits)

    beams = decoder.decode_beams(
        probs,
        beam_width=BEAM_WIDTH,
    )

    nbest_results = []

    for text, lm_state, timesteps, logit_score, lm_score in beams[:N_BEST]:
        scaled_ac_score = logit_score * ACOUSTIC_SCALE
        gpt2_score = compute_gpt2_score(text)
        final_rescored_total = scaled_ac_score + (2.0 * gpt2_score)

        nbest_results.append({
            'text': text,
            'ac_score': scaled_ac_score,
            'gpt2_score': gpt2_score,
            'total_score': final_rescored_total
        })

    best_hypothesis = min(nbest_results, key=lambda x: x['total_score'])

    print("Best Decoded Text:", best_hypothesis['text'])