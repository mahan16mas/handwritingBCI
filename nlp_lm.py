import pickle
import sys, os
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from pyctcdecode import build_ctcdecoder
from characterDefinitions import getHandwritingCharacterDefinitions
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--version', type=int, default=21)
args = parser.parse_args()
folder = args.folder
dataset_name = f"nlp{args.version}"
multi = 'multi'  in folder or (folder.startswith('256-dim'))
prefix = f"{dataset_name}_" if multi else ""
cache_dir = '/data/hossein/mm_project/cache/'
with open(f'/data/hossein/mm_project/speech_gru_cebra/nlp{args.version}.pkl', 'rb') as f:
    sentences = pickle.load(f)
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl", cache_dir=cache_dir)
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2-xl", cache_dir=cache_dir).to(device)
gpt_model.eval()

def compute_exact_gpt2_lm_rescore(text):
    if not text.strip():
        return 9999.0


    text_formatted = text[0].upper() + text[1:] if len(text) > 0 else text
    input_ids = gpt_tokenizer.encode(text_formatted)
    enc_text = [50256] + input_ids + [50256]
    tensor_input = torch.tensor([enc_text], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = gpt_model(tensor_input)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        log_sum = 0.0
        for t in range(1, len(enc_text)):
            target_token_id = enc_text[t]
            log_sum += log_probs[0, t - 1, target_token_id].item()

    lm_rescore = -log_sum
    return lm_rescore

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

charDef = getHandwritingCharacterDefinitions()
rootDir = "/data/hossein/mm_project" + '/handwritingBCIData/'
langModelDir = rootDir + 'BigramLM'

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
    length = (rnn_outputs["logitLengths"][i])
    logits = rnn_outputs["logits"][i][:length]
    sentence = sentences[i]
    trueText = sentence.replace('>', ' ')
    trueText = trueText.replace('~', '.')
    sentence = trueText.replace('#', '')
    logits = logits[:, charDef['idxToKaldi']]
    if not isinstance(logits, np.ndarray):
        logits = logits.numpy()
    probs = logits * ACOUSTIC_SCALE

    beams = decoder.decode_beams(
        logits=probs,
        beam_width=BEAM_WIDTH,
        beam_prune_logp=-10.0,
        token_min_logp=-10.0,
        prune_history=True
    )

    nbest_results = []

    for text, lm_state, timesteps, logit_score, lm_score in beams[:N_BEST]:
        scaled_ac_score = logit_score * ACOUSTIC_SCALE
        gpt2_score = compute_gpt2_score(text)
        final_rescored_total = -scaled_ac_score + (2.0 * gpt2_score)

        nbest_results.append({
            'text': text,
            'ac_score': scaled_ac_score,
            'gpt2_score': gpt2_score,
            'total_score': final_rescored_total
        })

    best_hypothesis = min(nbest_results, key=lambda x: x['total_score'])

    print("Best Decoded Text:", best_hypothesis['text'])
    print("True Sentence:", sentence)
    print("#"*20)