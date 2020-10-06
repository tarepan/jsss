# `npVCC2016` - Python loader of npVCC2016Corpus

`npVCC2016` is a Python loader set for [npVCC2016 non-parallel speech corpus](https://github.com/tarepan/npVCC2016Corpus).  
For machine learning, corpus/dataset is indispensable - but troublesome - part.  
We need portable & flexible loader for streamline development.  
`npVCC2016` is the one!!  

## Demo

Python/PyTorch  

```bash
pip install npVCC2016
```

```python
from npVCC2016.PyTorch.dataset.waveform import NpVCC2016

dataset = NpVCC2016(".", train=True, download=True)

for datum in dataset:
    print("Yeah, data is acquired with only two line of code!!")
    print(datum) # (datum, label) tuple provided
``` 

`npVCC2016` transparently downloads corpus, structures the data and provides standarized datasets.  
What you have to do is only instantiating the class!  

## APIs
Current `npVCC2016` support PyTorch.  
As interface, PyTorch's `Dataset` and PyTorch-Lightning's `DataModule` are provided.  
npVCC2016 corpus is speech corpus, so we provide `waveform` dataset and `spectrogram` dataset for both interfaces.  

- PyTorch
  - (pure PyTorch) dataset
    - waveform: `NpVCC2016`
    - spectrogram: `NpVCC2016_spec`
  - PyTorch-Lightning
    - waveform: `NpVCC2016DataModule`
    - spectrogram: `NpVCC2016_spec_DataModule`

### Extendibility
`waveform` dataset has easy-to-extend structure.  
By overiding hook functions, you can customize preprocessing for your machine-learning tasks.  
Please check `dataset`-`waveform` file.  