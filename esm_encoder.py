import os
import torch
from transformers import EsmModel, AutoTokenizer
from Bio import SeqIO
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
ESM_DIR = os.path.join(ROOT, "esm2_model")


def extract_esm_features(fasta_path, batch_size=1, max_len=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ESM] device: {device}")

    print(f"[ESM] loading model from {ESM_DIR}")
    model = EsmModel.from_pretrained(ESM_DIR)
    tokenizer = AutoTokenizer.from_pretrained(ESM_DIR)
    model = model.to(device).eval()

    seqs, ids = [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        if len(seq) > max_len:
            print(f"  truncating {record.id} ({len(seq)} -> {max_len})")
            seq = seq[:max_len]
        seqs.append(seq)
        ids.append(record.id)
    if not seqs:
        raise ValueError("empty FASTA")

    all_feat = []
    for i in tqdm(range(0, len(seqs), batch_size), desc="ESM"):
        batch = seqs[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        for j, s in enumerate(batch):
            L = len(s)
            all_feat.append(hidden[j, 1:L+1].mean(dim=0).cpu().numpy())

    feats = np.stack(all_feat)
    print(f"[ESM] shape: {feats.shape}")
    return feats, ids
