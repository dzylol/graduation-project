from src.data.molecule_dataset import random_split_dataset, get_next_split_seed

# Split ESOL
seed = get_next_split_seed()
train, val, test = random_split_dataset(
    "dataset/ESOL/delaney.csv", output_dir="dataset/ESOL/", seed=seed
)
print(f"ESOL Seed {seed}: Train={len(train)}, Val={len(val)}, Test={len(test)}")

# Split BBBP
seed = get_next_split_seed()
train, val, test = random_split_dataset(
    "dataset/BBBP/BBBP.csv", output_dir="dataset/BBBP/", seed=seed
)
print(f"BBBP Seed {seed}: Train={len(train)}, Val={len(val)}, Test={len(test)}")

# Split ClinTox
seed = get_next_split_seed()
train, val, test = random_split_dataset(
    "dataset/ClinTox/clintox.csv", output_dir="dataset/ClinTox/", seed=seed
)
print(f"ClinTox Seed {seed}: Train={len(train)}, Val={len(val)}, Test={len(test)}")
