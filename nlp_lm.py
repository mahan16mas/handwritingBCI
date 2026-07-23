import pickle
import sys, os
import numpy as np
from pyctcdecode import build_ctcdecoder

labels = ['', 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                 '>',',',"'",'~','?']
prefix = ''
rootDir = "/data/hossein/mm_project" + '/handwritingBCIData/'
langModelDir = rootDir+'BigramLM'
folder = "nlp_10_1_layer_11_5_days_20ms"
with open(f'/data/hossein/mm_project/speech_gru_cebra/{folder}/{prefix}logits', 'rb') as f:
    rnn_outputs = pickle.load(f)
for key in ["logits"]:
    new_list = []
    for item in rnn_outputs[key]:
        new_list.append(np.array(item))
    rnn_outputs[key] = new_list


decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path=langModelDir  + "/webTextSentences_tokenized-2gram-50000.arpa",
    alpha=0.5,
    beta=1.0
)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)
logits = rnn_outputs["logits"][0]
if not isinstance(logits, np.ndarray):
    logits = logits.numpy()
probs = softmax(logits)

decoded_text = decoder.decode(probs)
print(decoded_text)
