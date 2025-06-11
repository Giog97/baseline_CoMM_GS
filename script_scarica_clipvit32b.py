from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai/clip-vit-base-patch32",
    local_dir="./clip-vit-base-patch32"
)
