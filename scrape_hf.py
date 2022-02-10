import requests
import json
import os
from tqdm import tqdm
from huggingface_hub import HfApi, ModelFilter, ModelSearchArguments

import threading
import time

api = HfApi()
model_infos = api.list_models(
    fetch_config=True,
    full=True
)

for modelinfo in tqdm(model_infos[:2]):
    path_prefix = f"./models/{modelinfo.modelId}/"
    os.makedirs(path_prefix, exist_ok=True)

    with open(path_prefix+'infos.json', 'w') as fp:
        infos = modelinfo.__dict__
        infos["siblings"] = [s.__dict__ for s in infos["siblings"] if type(s) != dict]
        json.dump(infos, fp)

    r = requests.get(f"https://huggingface.co/{modelinfo.modelId}/raw/main/README.md", allow_redirects=True)
    open(path_prefix+'README.md', 'wb').write(r.content)

