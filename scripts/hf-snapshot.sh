#!/bin/bash

[ $# -ne 2 ] && echo "usage: $0 hf-model-id hf-token" && exit 1 

pip3 install --upgrade pip
pip3 install huggingface-hub==0.22.2

export SNAPSHOT_ROOT=$HOME/snapshots

export HF_MODEL_ID=$1
export HF_TOKEN=$2

LOG_ROOT=$HOME/logs/$HF_MODEL_ID

mkdir -p $LOG_ROOT
OUTPUT_LOG=$LOG_ROOT/hfsnapshot.log

SCRIPT=/tmp/hf_snapshot_${RANDOM}.py
cat > $SCRIPT <<EOF
from huggingface_hub import snapshot_download
from tempfile import TemporaryDirectory
from pathlib import Path
import shutil
import os

hf_model_id = os.environ.get("HF_MODEL_ID", None)
assert hf_model_id is not None, "HF_MODEL_ID must be set"
print(f"HF_MODEL_ID: {hf_model_id}")

hf_tensors = os.environ.get("HF_TENSORS", "true").lower() in ("true", "1")
print(f"Download Hugging Face Snapshot Tensors: {hf_tensors}")

hf_token = os.environ.get("HF_TOKEN", None)

print(f"Downloading HuggingFace snapshot: {hf_model_id}")
with TemporaryDirectory(suffix="model", prefix="hf", dir="/tmp") as cache_dir:
    ignore_patterns = ["*.msgpack", "*.h5"] if hf_tensors else [ "*.msgpack", "*.h5", "*.bin", "*.safetensors"]
    snapshot_download(repo_id=hf_model_id, 
        cache_dir=cache_dir,
        ignore_patterns=ignore_patterns,
        token=hf_token)

    cache_path = Path(cache_dir)
    local_snapshot_path = str(list(cache_path.glob(f"**/snapshots/*"))[0])
    print(f"Local snapshot path: {local_snapshot_path}")

    snapshot_path = os.path.join(os.environ["SNAPSHOT_ROOT"], os.environ["HF_MODEL_ID"])
    os.makedirs(snapshot_path, exist_ok=True)

    print(f"Copying snapshot to {snapshot_path}")
    for root, dirs, files in os.walk(local_snapshot_path):
        for file in files:
            full_path = os.path.join(root, file)
            if os.path.isdir(full_path):
                shutil.copytree(full_path, os.path.join(snapshot_path, os.path.basename(full_path)))
            else:
                shutil.copy2(full_path, snapshot_path)
            
EOF

python $SCRIPT 2>&1 | tee $OUTPUT_LOG
rm -f $SCRIPT