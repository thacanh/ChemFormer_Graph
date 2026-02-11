import torch
from molbart.models import BARTModel
from molbart.utils.tokenizers.tokenizers import ChemformerTokenizer
from molbart.utils.samplers import BeamSearchSampler

# Paths
MODEL = "./model/last.ckpt"
VOCAB = "../bart_vocab_downstream.json"

# Load tokenizer and model
print("Loading tokenizer...")
tokenizer = ChemformerTokenizer(VOCAB)

print("Creating sampler...")
sampler = BeamSearchSampler(tokenizer, [], 512, device="cpu", data_device="cpu")

print("Loading model from checkpoint...")
# Load BARTModel trực tiếp thay vì dùng Chemformer wrapper
# Cần truyền vocabulary_size và pad_token_idx như trong Chemformer._initialize_from_ckpt
model = BARTModel.load_from_checkpoint(
    MODEL, 
    decode_sampler=sampler, 
    vocabulary_size=len(tokenizer),
    pad_token_idx=tokenizer["pad"],
    map_location=torch.device('cpu')
)
model.eval()
model.to("cpu")

# Input product SMILES
product = "CC(=O)O"

print(f"\nProduct: {product}")
print("\nPredicting reactants...")

# Prepare batch manually
encoder_input = tokenizer.tokenize([product], pad=True)
encoder_pad_mask = encoder_input == tokenizer.pad_token_idx

# Create batch dictionary
batch = {
    "encoder_input": encoder_input.long(),
    "encoder_pad_mask": encoder_pad_mask,
    "target_smiles": [product]
}

# Predict using the model
with torch.no_grad():
    sampled_smiles, log_lhs = model.sample_molecules(
        batch,
        sampling_alg="beam"
    )

# Display predictions
print("\nPredicted reactants:")
for i, (smiles, log_lh) in enumerate(zip(sampled_smiles[0], log_lhs[0]), 1):
    print(f"{i}. {smiles} (log-likelihood: {log_lh:.4f})")
