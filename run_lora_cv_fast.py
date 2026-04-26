import subprocess
import os

# Only run 2 folds for speed
folds = [1, 2]
configs = [
    "configs/lora_config_r32.yaml",
    "configs/lora_config_r16.yaml",
    "configs/lora_config_r8.yaml"
]

for fold in folds:
    for config in configs:
        config_name = os.path.splitext(os.path.basename(config))[0]
        adapter_dir = f"data/models/lora_adapter/fold_{fold}/{config_name}/final_adapter"
        if os.path.isdir(adapter_dir):
            print(f"[SKIP] Already trained: {adapter_dir}")
            continue
        print(f"Training fold {fold} with config {config}...")
        subprocess.run([
            "python", "scripts/train_lora.py",
            "--fold", str(fold),
            "--config", config
        ])
