import os
import shutil
from huggingface_hub import HfApi

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise Exception("HF_TOKEN environment variable not set!")

api = HfApi()

output_dir = "/mnt"

# Find all experiment_run_* folders
runs = [
    name for name in os.listdir(output_dir)
    if name.startswith("experiment_run") and os.path.isdir(os.path.join(output_dir, name))
]

if not runs:
    print("⚠️ No experiment_run folders found in /mnt/. Exiting.")
    exit(1)

for run in runs:
    run_path = os.path.join(output_dir, run)
    weights_path = os.path.join(run_path, "model", "model_weights")

    if not os.path.exists(weights_path):
        print(f"⚠️ Weights path {weights_path} does not exist for {run}, skipping.")
        continue

    # 📂 Copy weights to root of experiment_run
    print(f"📄 Copying weights from {weights_path} to {run_path}...")
    for filename in os.listdir(weights_path):
        src = os.path.join(weights_path, filename)
        dst = os.path.join(run_path, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    print(f"✅ Weights copied to {run_path}.")

    repo_id = f"Latish4522/beyonce-adapter-{run[-6:]}"

    # 🚀 Create repo if needed
    try:
        api.create_repo(
            repo_id=repo_id,
            token=hf_token,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ Repo {repo_id} ready.")
    except Exception as e:
        print(f"⚠️ Failed to create repo {repo_id}: {e}")
        continue

    # ⬆️ Upload entire experiment_run folder
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=run_path,
            token=hf_token,
            repo_type="model",
            commit_message=f"Upload full run {run}"
        )
        print(f"✅ Successfully uploaded full run {run} to HuggingFace.")

    except Exception as e:
        print(f"❌ Failed to upload {run}: {e}")