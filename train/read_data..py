import pickle
from rdkit import Chem

# ===== load pickle =====
with open("data/uspto_50.pickle", "rb") as f:
    df = pickle.load(f)

print("Loaded dataframe:", df.shape)

# ===== Mol → SMILES =====
df["reactants_smiles"] = df["reactants_mol"].apply(
    lambda m: Chem.MolToSmiles(m) if m is not None else ""
)

df["products_smiles"] = df["products_mol"].apply(
    lambda m: Chem.MolToSmiles(m) if m is not None else ""
)

# ===== save CSV =====
df_out = df[["reactants_smiles", "products_smiles", "set"]]
df_out.to_csv("uspto_smiles.csv", index=False)

print("Saved → uspto_smiles.csv")
print(df_out.head())
