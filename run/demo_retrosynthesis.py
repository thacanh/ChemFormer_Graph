import torch
import math

from molbart.models import BARTModel
from molbart.utils.tokenizers.tokenizers import ChemformerTokenizer
from molbart.utils.samplers import BeamSearchSampler
from molbart.data.util import BatchEncoder


MODEL_PATH = "./model/last.ckpt"
VOCAB_PATH = "../bart_vocab.json"
DEVICE = "cpu"
N_BEAMS = 5
MAX_LEN = 512


TEST_PRODUCTS = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Acetic acid", "CC(=O)O"),
    ("Ethanol", "CCO"),
    ("Benzene", "c1ccccc1"),
    ("Acetone", "CC(=O)C"),
]


def load_model():
    tokenizer = ChemformerTokenizer(filename=VOCAB_PATH)
    encoder = BatchEncoder(tokenizer, None, MAX_LEN)

    sampler = BeamSearchSampler(
        tokenizer,
        [],
        MAX_LEN,
        device=DEVICE,
        data_device=DEVICE
    )

    model = BARTModel.load_from_checkpoint(
        MODEL_PATH,
        decode_sampler=sampler,
        vocabulary_size=len(tokenizer),
        pad_token_idx=tokenizer["pad"],
        map_location=torch.device(DEVICE)
    )

    model.eval()
    model.to(DEVICE)

    return model, encoder


def predict(model, encoder, smiles):
    encoder_input, encoder_pad_mask = encoder([smiles])

    batch = {
        "encoder_input": encoder_input,
        "encoder_pad_mask": encoder_pad_mask,
        "target_smiles": [smiles]
    }

    with torch.no_grad():
        sampled_smiles, log_lhs = model.sample_molecules(batch, sampling_alg="beam")

    return sampled_smiles[0], log_lhs[0]


def main():
    model, encoder = load_model()

    for name, smiles in TEST_PRODUCTS:
        print("\n")
        print(name)
        print("Product:", smiles)

        preds, scores = predict(model, encoder, smiles)

        for i, (p, s) in enumerate(zip(preds, scores), 1):
            conf = math.exp(float(s)) * 100
            print(f"{i:2d}. {p}")
            print(f"    logP = {s:.4f} | confidence ~ {conf:.2f}%")


if __name__ == "__main__":
    main()
