#!/bin/bash
cd /home/qfh/graduation-project

echo "=== Splitting ESOL ==="
/usr/bin/podman run --rm -v "$(pwd):/workspace" --workdir /workspace localhost/bimamba bash -c "cd /workspace && python -c 'from src.data.molecule_dataset import random_split_dataset, get_next_split_seed; seed = get_next_split_seed(); train, val, test = random_split_dataset(\"dataset/ESOL/delaney.csv\", output_dir=\"dataset/ESOL/\", seed=seed); print(f\"ESOL Seed {seed}: Train={len(train)}, Val={len(val)}, Test={len(test)}\")'"

echo "=== Splitting BBBP ==="
/usr/bin/podman run --rm -v "$(pwd):/workspace" --workdir /workspace localhost/bimamba bash -c "cd /workspace && python -c 'from src.data.molecule_dataset import random_split_dataset, get_next_split_seed; seed = get_next_split_seed(); train, val, test = random_split_dataset(\"dataset/BBBP/BBBP.csv\", output_dir=\"dataset/BBBP/\", seed=seed); print(f\"BBBP Seed {seed}: Train={len(train)}, Val={len(val)}, Test={len(test)}\")'"

echo "=== Splitting ClinTox ==="
/usr/bin/podman run --rm -v "$(pwd):/workspace" --workdir /workspace localhost/bimamba bash -c "cd /workspace && python -c 'from src.data.molecule_dataset import random_split_dataset, get_next_split_seed; seed = get_next_split_seed(); train, val, test = random_split_dataset(\"dataset/ClinTox/clintox.csv\", output_dir=\"dataset/ClinTox/\", seed=seed); print(f\"ClinTox Seed {seed}: Train={len(train)}, Val={len(val)}, Test={len(test)}\")'"

echo "=== Done ==="
