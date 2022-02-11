import requests
import json
import os
from tqdm import tqdm
from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

num_workers = 40

def scrape_model(modelinfo, pbar):
    pbar.update(1)
    path_prefix = f"./models/{modelinfo.modelId}/"

    if os.path.exists(path_prefix) and os.path.exists(path_prefix + "README.md"):
        # delete README if empty
        with open(path_prefix + "README.md", "r") as f:
            if f.read() == "Error: the path 'README.md' does not exist in the given tree":
                os.remove(path_prefix + "README.md")
        return
    os.makedirs(path_prefix, exist_ok=True)

    with open(path_prefix+'infos.json', 'w') as fp:
        infos = modelinfo.__dict__
        infos["siblings"] = [s.__dict__ for s in infos["siblings"] if type(s) != dict]
        json.dump(infos, fp)

    r = requests.get(f"https://huggingface.co/{modelinfo.modelId}/raw/main/README.md", allow_redirects=True)
    open(path_prefix+'README.md', 'wb').write(r.content)


if __name__ == "__main__":
    api = HfApi()
    model_infos = api.list_models(
        fetch_config=True,
        full=True
    )

    with tqdm(total=len(model_infos)) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(scrape_model, model, pbar) for model in model_infos]
            for future in as_completed(futures):
                result = future.result()