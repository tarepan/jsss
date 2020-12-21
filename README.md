# jsss - Python loader of JSSS
<!-- [![PyPI version](https://badge.fury.io/py/npvcc2016.svg)](https://badge.fury.io/py/npVCC2016) -->
<!-- ![Python Versions](https://img.shields.io/pypi/pyversions/npvcc2016.svg)   -->

`jsss` is a Python loader of **[JSSS: Japanese speech corpus 
for summarization and simplification](https://sites.google.com/site/shinnosuketakamichi/research-topics/jsss_corpus)**.  
For machine learning, corpus/dataset is indispensable - but troublesome - part.  
We need portable & flexible loader for streamline development.  
`jsss` is the one!  

## Demo

Python/PyTorch  

<!-- ```bash
pip install npvcc2016
```

```python
from npvcc2016.PyTorch.dataset.waveform import NpVCC2016

dataset = NpVCC2016(".", train=True, download=True)

for datum in dataset:
    print("Yeah, data is acquired with only two line of code!!")
    print(datum) # (datum, label) tuple provided
```  -->

`jsss` transparently downloads corpus, structures the data and provides standarized datasets.  
What you have to do is only instantiating the class!  

## APIs
Current `jsss` support PyTorch.  
As interface, PyTorch's `Dataset` and PyTorch-Lightning's `DataModule` are provided.  
jsss corpus is speech corpus, so we provide `waveform` dataset and `spectrogram` dataset for both interfaces.  

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