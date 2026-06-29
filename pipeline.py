import argparse, os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

from esm_encoder import extract_esm_features
from pex1_regressor import predict_with_model
from pex3_voting import VotingXGBRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "PEXn_model")
TARGETS = ["Topt", "Tm", "pH"]

PEX1_FILES = {
    t: (f"PEX1_{t}.model.joblib", f"PEX1_{t}.scaler.joblib") for t in TARGETS
}
PEX3_FILES = {
    t: (f"PEX3_{t}.model.joblib", f"PEX3_{t}.scaler.joblib") for t in TARGETS
}


def load_pex3_group():
    models, scalers = {}, {}
    for t in TARGETS:
        fn, fs = PEX3_FILES[t]
        models[t] = joblib.load(os.path.join(MODEL_DIR, fn))
        scalers[t] = joblib.load(os.path.join(MODEL_DIR, fs))
        print(f"  [PEX3] {t.upper()}")
    return models, scalers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 60)
    print("  PEXnToptv2 — ESM2-t33 + XGBoost pipeline")
    print("=" * 60)

    print("\n[1/3] ESM2-t33 feature extraction")
    feats, ids = extract_esm_features(args.fasta, args.batch_size, args.max_len)

    print("\n[2/3] loading models")
    print("  [PEX1] single XGBoost (via pex1_regressor)")
    print("  [PEX3] voting XGBoost (via pex3_voting)")
    m3, s3 = load_pex3_group()

    print("\n[3/3] predicting")
    res = {"sequence_id": ids}
    for t in TARGETS:
        mp = os.path.join(MODEL_DIR, PEX1_FILES[t][0])
        sp = os.path.join(MODEL_DIR, PEX1_FILES[t][1])
        res[f"PEX1{t}"] = np.round(predict_with_model(mp, sp, feats), 4)
        X = s3[t].transform(feats)
        res[f"PEX3{t}"] = np.round(m3[t].predict(X), 4)

    df = pd.DataFrame(res)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nDone -> {args.output}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
