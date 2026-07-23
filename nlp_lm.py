import pickle
import sys, os
import numpy as np
from pyctcdecode import build_ctcdecoder
from characterDefinitions import getHandwritingCharacterDefinitions
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
langModelDir = rootDir+'BigramLM'
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
    kenlm_model_path=langModelDir  + "/webTextSentences_tokenized-2gram-50000.arpa",
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
        kaldi_total_score = scaled_ac_score + (2.0 * lm_score)

        nbest_results.append({
            'text': text,
            'ac_score': scaled_ac_score,
            'lm_score': lm_score,
            'total_score': kaldi_total_score
        })

    best_hypothesis = min(nbest_results, key=lambda x: x['total_score'])

    print("Best Decoded Text:", best_hypothesis['text'])
