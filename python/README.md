# SignSpeak — Python training pipeline

## Install
```bash
pip install -r requirements.txt
```

## 1. Collect dataset
```bash
python collect_data.py
```
Records 30 samples × 30 frames per class from your webcam. Press Enter to
start each take. Output: `dataset/<class>/<idx>/sequence.npy` shaped
`(30, 126)` — 21 landmarks × 3 coords × 2 hands.

## 2. Train on Google Colab (recommended)
Upload `dataset/` and `train_lstm.py` to a Colab notebook with a T4 GPU runtime:
```bash
!pip install -q tensorflow scikit-learn
!python train_lstm.py
```
~15–30 min for 100 classes. Produces `best_model.keras` and `labels.npy`.

## 3. Export for the browser
```bash
pip install tensorflowjs
python export_tfjs.py
```

## 4. Deploy into the React app
```
tfjs_model_quantized/model.json          → public/model/model.json
tfjs_model_quantized/group1-shard*.bin   → public/model/
labels.json                              → public/labels.json
```
Reload the page — the demo banner disappears and live inference takes over.

## Train from Kaggle datasets (WLASL + ISL/CSLTR)
Install Kaggle helper first:
```bash
pip install kagglehub
```

Download datasets:
```python
import kagglehub
wlasl_path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
isl_path = kagglehub.dataset_download("drblack00/isl-csltr-indian-sign-language-dataset")
print(wlasl_path, isl_path)
```

Convert them into this project's sequence format:
```bash
python build_dataset_from_kaggle.py   --dataset-roots "$WLASL_PATH" "$ISL_PATH"   --output dataset   --windows-per-class 100   --max-videos-per-class 100
```

Or run one command (auto-download inside the script; no manual download step):
```bash
python build_dataset_from_kaggle.py \
  --kaggle-datasets risangbaskoro/wlasl-processed drblack00/isl-csltr-indian-sign-language-dataset \
  --output dataset \
  --windows-per-class 100 \
  --max-videos-per-class 100
```

Then train as usual:
```bash
python train_lstm.py
```
