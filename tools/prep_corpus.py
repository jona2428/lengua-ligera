"""Prepara el corpus: Wikipedia en español -> ids de caracteres (u8).

Salida en data/:
  train.bin, valid.bin  bytes = id de carácter
  vocab.txt             un carácter por línea (escapado como JSON), id = línea
"""
import json, random, re, collections
import pyarrow.parquet as pq

SRC = "data/wiki_es_10.parquet"
TRAIN_CHARS = 100_000_000
VALID_CHARS = 2_000_000
COVERAGE = 0.9999
CUT = re.compile(r"\n(Referencias|Enlaces externos|Bibliografía|Véase también|Notas|Fuentes)\s*\n")

texts = pq.read_table(SRC, columns=["text"]).column("text").to_pylist()
docs = []
for t in texts:
    m = CUT.search(t)
    if m:
        t = t[: m.start()]
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) >= 500:
        docs.append(t)
random.Random(0).shuffle(docs)

valid, train, n = [], [], 0
for d in docs:
    if n < VALID_CHARS:
        valid.append(d); n += len(d) + 2
    elif n < VALID_CHARS + TRAIN_CHARS:
        train.append(d); n += len(d) + 2
    else:
        break
train_s = "\n\n".join(train)
valid_s = "\n\n".join(valid)

freq = collections.Counter(train_s)
total = sum(freq.values())
vocab, acc = [], 0
for ch, c in freq.most_common():
    vocab.append(ch); acc += c
    if acc / total >= COVERAGE or len(vocab) == 127:
        break
UNK = "�"
vocab.append(UNK)
idx = {c: i for i, c in enumerate(vocab)}
unk = idx[UNK]

def enc(s):
    return bytes(idx.get(c, unk) for c in s)

open("data/train.bin", "wb").write(enc(train_s))
open("data/valid.bin", "wb").write(enc(valid_s))
with open("data/vocab.txt", "w", encoding="utf-8") as f:
    for c in vocab:
        f.write(json.dumps(c) + "\n")
unk_rate = sum(1 for c in valid_s if c not in idx) / len(valid_s)
print(f"docs {len(docs)}  train {len(train_s)/1e6:.1f}M  valid {len(valid_s)/1e6:.2f}M  vocab {len(vocab)}  unk valid {unk_rate:.5f}")
