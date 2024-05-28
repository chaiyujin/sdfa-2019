# Speech-Driven Facial Animation with Spectral Gathering and Temporal Attention

[Project Website](https://chaiyujin.github.io/sdfa)

## Install dependencies
Necessary libraries:
```bash
# install python libs
$ python3 -m pip install -r requirements.txt
# install cmake and sndfile lib
$ sudo apt install libsndfile1 cmake
```

(not necessary) If you want to prepare dataset, [montreal-forced-aligner](https://montreal-forced-aligner.readthedocs.io/) must be installed. <small>*(Some errors may occur during installation, please pay attention.)*</small>
```bash
$ bash scripts/install_mkl.sh
$ bash scripts/install_kaldi.sh
$ bash scripts/install_mfa.sh
```

## Evaluate
Download pretrained model from [Google Drive](https://drive.google.com/file/d/1x5srFdb48BFmkE04AAdOGguj9ghG2PAw/view?usp=sharing), unzip it, and put in `./pretrained_models/dgrad`.

Modify and run evaluate script `bash evaluate.sh`.

## Prepare VOCASET
Download VOCASET from https://voca.is.tue.mpg.de/
Unzip directories:
```
| VOCASET
 -| unposedcleaneddata
 -| sentencestext
 -| templates
 -| audio
```
Run the preload python script.
```
python3 -m saberspeech.datasets.voca.preload\
    --source_root <ROOT_VOCASET> \
    --output_root <ROOT_PROCESSED>
```

## Pre-trained models
- [ ] dgrad
- [ ] offsets
- [ ] PCA of dgrad, offsets

## Citation
```
@article{chai2022speech,
  title={Speech-driven facial animation with spectral gathering and temporal attention},
  author={Chai, Yujin and Weng, Yanlin and Wang, Lvdi and Zhou, Kun},
  journal={Frontiers of Computer Science},
  volume={16},
  number={3},
  pages={1--10},
  year={2022},
  publisher={Springer}
}
```
