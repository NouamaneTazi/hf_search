# Huggingface Semantic Search Engine

This library uses the [Retrieve & Re-Rank Pipeline](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) to search for models on HuggingFace, based on their READMEs, and returns their metadata using the [`huggingface_hub` library](https://huggingface.co/docs/hub/searching-the-hub)

This project is largely inspired by [Nils Reimers](https://www.nils-reimers.de/)' work so please make sure to check out [his library](https://www.sbert.net/index.html).

### Installation

```bash
pip install git+<https://github.com/NouamaneTazi/hf_search>
```

### Usage

* Search for models that can transcribe french audios

> You can compare with other search methods supported (`bm25`, `retrieve` or `retrieve & rerank`)

```python
>> from search import HFSearch
>> hf_search = HFSearch("path/to/hf_data", "path/to/embeddings/multi-qa-MiniLM-L6-cos-v1-embeddings.pt")
>> search(query="transcribe french audio", method="retrieve & rerank", limit=3)
  
  [{'passage': '### Transcribing your own audio files (in French)\n', # most relevant passage
  'modelId': 'speechbrain/asr-wav2vec2-commonvoice-fr', 
  'sha': '33b4e3ab46e2787406d65e41c1473e42506cb901',
  'lastModified': '2021-12-18T09:12:59.000Z',
  'tags': ['wav2vec2',
   'feature-extraction',
   'fr',
   'dataset:commonvoice',
   'speechbrain',
   'CTC',
   'pytorch',
   'Transformer',
   'license:apache-2.0',
   'automatic-speech-recognition'],
  'pipeline_tag': 'automatic-speech-recognition',
  'siblings': [{'rfilename': '.gitattributes'},
   {'rfilename': 'README.md'},
   {'rfilename': 'asr.ckpt'},
   {'rfilename': 'config.json'},
   {'rfilename': 'example-fr.wav'},
   {'rfilename': 'example.wav'},
   {'rfilename': 'hyperparams.yaml'},
   {'rfilename': 'preprocessor_config.json'},
   {'rfilename': 'tokenizer.ckpt'},
   {'rfilename': 'wav2vec2.ckpt'}],
  'config': {'architectures': ['Wav2Vec2Model'],
   'model_type': 'wav2vec2',
   'speechbrain': {'interface': 'EncoderASR'}},
  'private': False,
  'library_name': 'speechbrain',
  'likes': 1,
  'readme': '---\nlanguage: "fr"\nthumbnail:\npipeline_tag:...', # truncated here for demo
  'downloads': 192.0,
  'score': 5.644802}, # relevance score
  ...
  ]
```

* Filter the results by the model's task (`pipeline_tag`) and library, like described in [here](https://huggingface.co/docs/hub/searching-the-hub#searching-for-a-model)

```python
>> search(query="model that detects birds", method="retrieve & rerank", limit=3, filters={"task": ['automatic_speech_recognition', 'speech_processing', 'other'], "library_name": ['transformers', 'PyTorch']})
```

* Sort the results by `lastModified`, `likes` or `downloads` (by default it's by relevance). You can also sort by ascending order by setting `direction = "ascending"`

```python
>> search(query="model that detects birds", limit=3, sort="downloads", direction="ascending")
```

### How does it work?

![](https://raw.githubusercontent.com/UKPLab/sentence-transformers/master/docs/img/InformationRetrieval.png)
Given a search query, we first use a **retrieval system** that retrieves a large list of e.g. 100 possible hits which are potentially relevant for the query. For the retrieval, we can use either lexical search, e.g. with ElasticSearch, or we can use dense retrieval with a bi-encoder.

However, the retrieval system might retrieve documents that are not that relevant for the search query. Hence, in a second stage, we use a re-ranker based on a cross-encoder that scores the relevancy of all candidates for the given search query.

The output will be a ranked list of hits we can present to the user.

[Read more about it in here.](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)

### Data

The library extracts model's metadata from a folder `hf_data`

```
hf_data
├── models.jsonl
└── passages.jsonl
```

where `passages.jsonl` contain the passages to be encoded

```json
{"id":0,"modelId":"tizaino\/bert-base-uncased-finetuned-Pisa","passage":"## Model description\n"}
{"id":0,"modelId":"tizaino\/bert-base-uncased-finetuned-Pisa","passage":"More information needed\n"}
{"id":0,"modelId":"tizaino\/bert-base-uncased-finetuned-Pisa","passage":"## Intended uses & limitations\n"}
{"id":0,"modelId":"tizaino\/bert-base-uncased-finetuned-Pisa","passage":"More information needed\n"}
{"id":0,"modelId":"tizaino\/bert-base-uncased-finetuned-Pisa","passage":"## Training and evaluation data\n"}
{"id":0,"modelId":"tizaino\/bert-base-uncased-finetuned-Pisa","passage":"More information needed\n"}
...

```

and `models.jsonl` contains the metadata for each model

```json
{
  "modelId": "hf-test/xls-r-dummy",
  "sha": "ed3e4d304b193c575f8de763563b55888520c08c",
  "lastModified": "2022-01-09T00:32:41.000Z",
  "tags": ["pytorch", "wav2vec2", "feature-extraction", "transformers"],
  "pipeline_tag": "feature-extraction",
  "siblings": [
    { "rfilename": ".gitattributes" },
    { "rfilename": "config.json" },
    { "rfilename": "preprocessor_config.json" },
    { "rfilename": "pytorch_model.bin" }
  ],
  "config": { "architectures": ["Wav2Vec2Model"], "model_type": "wav2vec2" },
  "private": false,
  "library_name": "transformers",
  "likes": 0,
  "readme": null,
  "downloads": 317
}
...
```
