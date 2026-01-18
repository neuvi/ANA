#!/usr/bin/env python3
"""Download rerank model to local data folder.

Usage:
    python -m scripts.download_rerank_model
    # or
    ana model download rerank
"""

from pathlib import Path
import sys

def download_rerank_model(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    target_dir: Path = Path("data/models/rerank")
) -> Path:
    """Download rerank model to local directory.
    
    Args:
        model_name: HuggingFace model name
        target_dir: Target directory for the model
        
    Returns:
        Path to the downloaded model
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("❌ sentence-transformers not installed.")
        print("   Run: uv add sentence-transformers")
        sys.exit(1)
    
    # Create target directory
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Model folder name (replace / with _)
    model_folder = model_name.replace("/", "_")
    model_path = target_dir / model_folder
    
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return model_path
    
    print(f"📥 Downloading {model_name}...")
    print(f"   Target: {model_path}")
    
    # Download and load model
    model = CrossEncoder(model_name)
    
    # Save to local directory
    model.save(str(model_path))
    
    print(f"✅ Model saved to: {model_path}")
    return model_path


if __name__ == "__main__":
    download_rerank_model()
