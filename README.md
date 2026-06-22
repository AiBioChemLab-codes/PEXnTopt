# Protein Property Prediction Pipeline

End-to-end prediction of enzyme properties (Topt, Tm, optimal pH) from protein FASTA sequences using ESM2-t33 features and XGBoost regression.

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

Install:
```bash
pip install -r requirements.txt
```

## ⚠️ Important: Download ESM2 Model

The model weights are **not** included in this repository due to size (>2.5GB).
You must download them manually to the `esm2_model/` folder.

Run this command in the project root:
```bash
huggingface-cli download facebook/esm2_t33_650M_UR50D --local-dir ./esm2_model
```

If `huggingface-cli` is not available, install it first:
```bash
pip install huggingface_hub
```

After download, the folder should contain `pytorch_model.bin`, `config.json`, etc.

## Usage

```bash
python pipeline.py --fasta <input.fasta> --output <results.csv>
```

Example:
```bash
python pipeline.py --fasta test.fasta --output predictions.csv
```

Output columns: `PEX1_opt`, `PEX1_tm`, `PEX1_ph`, `PEX3_opt`, `PEX3_tm`, `PEX3_ph`.

## Notes

- GPU recommended but CPU works (slower).
- Sequences longer than 1000 amino acids are truncated.
- Adjust `--batch_size` if memory issues occur.

For more details, refer to the source code comments.
