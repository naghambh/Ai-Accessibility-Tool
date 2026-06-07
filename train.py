# train.py
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW


# -----------------------
# 1. LOAD ALL DATA PAIRS
# -----------------------
def load_all_pairs(orig_path, simp_pattern):
    pairs = []
    
    # Load original sentences
    with open(orig_path, "r", encoding="utf-8") as f:
        orig_lines = f.readlines()

    # Loop over ALL simplification files: simp.0, simp.1, ...
    for simp_file in glob.glob(simp_pattern):
        with open(simp_file, "r", encoding="utf-8") as f:
            simp_lines = f.readlines()

        # Pair each original with its simplified version
        for o, s in zip(orig_lines, simp_lines):
            pairs.append((o.strip(), s.strip()))

    print(f"✅ Loaded {len(pairs)} sentence pairs.")
    return pairs

# -----------------------
# 2. DATASET CLASS
# -----------------------
class TextDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        original, simplified = self.pairs[idx]

        input_ids = self.tokenizer.encode(
            original,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        ).squeeze()

        target_ids = self.tokenizer.encode(
            simplified,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        ).squeeze()

        return {
            "input_ids": input_ids,
            "labels": target_ids
        }

# -----------------------
# 3. TRAIN LOOP
# -----------------------
def train_model(dataset):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=3e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Training on: {device}")
    model.to(device)

    epochs = 1  # Start small
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1} completed. Loss: {total_loss:.4f}")

    # Save model
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
    print("✅ Model saved to /model")

# -----------------------
# 4. MAIN
# -----------------------
if __name__ == "__main__":
    pairs = load_all_pairs(
        "dataset/asset.test.orig",
        "dataset/asset.test.simp.*"
    )

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    dataset = TextDataset(pairs, tokenizer)

    train_model(dataset)
