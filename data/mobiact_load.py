from pathlib import Path
import shutil

import kagglehub


def download_to_project_data(dataset_id: str, folder_name: str) -> Path:
    source_path = Path(kagglehub.dataset_download(dataset_id))
    target_path = Path(__file__).resolve().parent / "raw" / folder_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    return target_path


path = download_to_project_data("kmknation/mobifall-dataset-v20", "MOBIACT_Dataset")
print("Dataset saved to:", path)
