# jsss - Python loader of JSSS
<!-- [![PyPI version](https://badge.fury.io/py/npvcc2016.svg)](https://badge.fury.io/py/npVCC2016) -->
<!-- ![Python Versions](https://img.shields.io/pypi/pyversions/npvcc2016.svg)   -->

`jsss` is a Python loader of **[JSSS: Japanese speech corpus 
for summarization and simplification](https://sites.google.com/site/shinnosuketakamichi/research-topics/jsss_corpus)**.  
For machine learning, corpus/dataset is indispensable - but troublesome - part.  
We need portable & flexible loader for streamline development.  
`jsss` is the one!  

## Demo
### Python - Access an item
Get corpus item *path* without thinking about corpus handling.  

```python
# !pip install git+https://github.com/tarepan/jsss
from jsss.corpus import JSSS

# Get corpus archive and extract contents transparently.
corpus = JSSS(download_origin=True)
corpus.get_contents()
# > 32%|██████████                      | 0.33G/1.01G [00:42<01:31, 223MB/s]

# Get item IDs.
items = corpus.get_identities()

# Get path of each items.
for id in items:
  path = corpus.get_item_path(id)
  print(path)
  # > data/corpuses/JSSS/contents/jsss_ver1/short-form/basic5000/wav24kHz16bit/BASIC5000_0001.wav
  # > data/corpuses/JSSS/contents/jsss_ver1/short-form/basic5000/wav24kHz16bit/BASIC5000_0002.wav
  # ...
```
### Python/PyTorch - 1-lile Dataset
Get waveform dataset for PyTorch with 1-line!  

```python
# !pip install git+https://github.com/tarepan/jsss
from jsss.PyTorch.dataset.waveform import JSSS_wave

# That's all!
dataset = JSSS_wave(download_corpus=True)

for datum in dataset:
    print("Yeah, data is acquired with only single line of code!!")
    print(datum)
    # > Datum_JSSS_wave(waveform=tensor([-0.0007, -0.0007, -0.0008, ...
```

## APIs
Current `jsss` support PyTorch.  
As interface, PyTorch's `Dataset` and PyTorch-Lightning's `DataModule` are provided.  
JSSS corpus is speech corpus, so we provide `waveform` dataset and `spectrogram` dataset for both interfaces.  

- PyTorch
  - (pure PyTorch) dataset
    - waveform: `JSSS_wave`
    - spectrogram: `JSSS_spec`
  - PyTorch-Lightning
    - waveform: `JSSSDataModule`
    - spectrogram: `JSSS_spec_DataModule`

## Original paper
[![Paper](http://img.shields.io/badge/paper-arxiv.2010.01793-B31B1B.svg)][paper]  
<!-- https://arxiv2bibtex.org/?q=2010.01793&format=bibtex -->
```
@misc{2010.01793,
Author = {Shinnosuke Takamichi and Mamoru Komachi and Naoko Tanji and Hiroshi Saruwatari},
Title = {JSSS: free Japanese speech corpus for summarization and simplification},
Year = {2020},
Eprint = {arXiv:2010.01793},
}
```

## Dependency Notes
### PyTorch version
PyTorch version: PyTorch v1.6 is working (We checked with v1.6.0).  

For dependency resolution, we do **NOT** explicitly specify the compatible versions.  
PyTorch have several distributions for various environment (e.g. compatible CUDA version.)  
Unfortunately it make dependency version management complicated for dependency management system.  
In our case, the system `poetry` cannot handle cuda variant string (e.g. `torch>=1.6.0` cannot accept `1.6.0+cu101`.)  
In order to resolve this problem, we use `torch==*`, it is equal to no version specification.  
`Setup.py` could resolve this problem (e.g. `torchaudio`'s `setup.py`), but we will not bet our effort to this hacky method.  


[paper]:https://arxiv.org/abs/2010.01793