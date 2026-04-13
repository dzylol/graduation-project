from src.data.molecule_dataset import random_split_dataset, get_next_split_seed
seed = get_next_split_seed()
train, val, test = random_split_dataset("dataset/ESOL/delaney.csv", output_dir="dataset/ESOL/", seed=seed)
print(f"ESOL Seed {seed}: Train={len(train)}, Val={len(val)}, Test={len(test)}")
