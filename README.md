# sdfa-2019

## TODO
- [ ] experiments/pca
- [ ] Clean viewer
- [ ] Dump mesh of each frame
- [ ] Render with blender


## Install dependencies
Necessary libraries:
```bash
# install python libs
$ python3 -m pip install -r requirements.txt
# install cmake and sndfile lib
$ sudo apt install libsndfile1 cmake
```
If you want to prepare dataset, [montreal-forced-aligner](https://montreal-forced-aligner.readthedocs.io/) must be installed. <small>*(Some errors may occur during installation, please pay attention.)*</small>
```bash
$ bash scripts/install_mkl.sh
$ bash scripts/install_kaldi.sh
$ bash scripts/install_mfa.sh
```


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
