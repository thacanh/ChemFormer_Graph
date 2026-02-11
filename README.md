# ChemFormer Graph

## 1. Clone the repository

```bash
git clone https://github.com/thacanh/ChemFormer_Graph.git
cd ChemFormer_Graph
```

---

## 2. Create conda environment

```bash
conda env create -f env-dev.yml
```

## 3. Activate environment

```bash
conda activate chemformer
```

---

# Train from scratch

Run:

```bash
python train/train_newchemformer.py
```

---

# Finetune from pretrained model

## Step 1: Download pretrained model

Download from:
[https://drive.google.com/file/d/1Bt_6XnlPq2IJ3HvANUZfhU4zDLU5wYkp/view?usp=sharing](https://drive.google.com/file/d/1Bt_6XnlPq2IJ3HvANUZfhU4zDLU5wYkp/view?usp=sharing)

## Step 2: Place the checkpoint

After downloading, rename the file to:

```
last.ckpt
```

Move it into:

```
train/model/
```

Folder structure:

```
ChemFormer_Graph
 └── train
     └── model
         └── last.ckpt
```

## Step 3: Run finetuning

```bash
python train/finetune_newchemformer.py
```

---

# Notes

* Make sure the `chemformer` environment is activated before running any script.
* If you encounter missing package errors:

```bash
pip install -e .
```

---

# Quick start (full flow)

```bash
git clone https://github.com/thacanh/ChemFormer_Graph.git
cd ChemFormer_Graph

conda env create -f env-dev.yml
conda activate chemformer

# Train
python train/train_newchemformer.py

# Finetune
# download model -> put into train/model/last.ckpt
python train/finetune_newchemformer.py
```
