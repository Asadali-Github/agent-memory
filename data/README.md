# Data (gitignored — only this README is committed)

## LongMemEval (Mode A)

1. Get `longmemeval_s.json` from the official release
   (https://github.com/xiaowu0162/LongMemEval). On your machine:

   ```bash
   pip install huggingface_hub
   hf download xiaowu0162/longmemeval longmemeval_s.json --repo-type dataset --local-dir data/
   ```

   (If the file is shipped compressed, e.g. `.tar.gz`, extract so the JSON
   sits at the path below. The GitHub README's Google Drive link is the
   fallback if HF layout changes.)
2. Place it at `data/longmemeval_s.json`.
3. The capped, stratified subset 