## Unsupervised VoxCeleb trainer

This repository contains the training code for the models described in the paper '_Augmentation adversarial training for self-supervised speaker recognition_'.

#### Dependencies
```
pip install -r requirements.txt
```

#### Data preparation

The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets are used for these experiments. Follow the instructions on [this page](https://github.com/clovaai/voxceleb_trainer) to download and prepare the data for training.

In addition, you need to download the [MUSAN](https://www.openslr.org/17/) noise corpus. 

First, download and extract the files, then use the command

```
python ./process_musan.py /parent/dir/of/musan/
```

to split the audio files into short segments for faster random access.

#### Training example

```
python ./trainSpeakerNet.py --log_input True --save_path data/exp1 --augment_anchor True --augment_type 2 --train_list /path/to/voxcelebs/train_list.txt  --test_list /path/to/voxcelebs/test_list.txt --train_path /path/to/voxcelebs/voxceleb2 --test_path /path/to/voxcelebs/voxceleb1 --musan_path /path/to/musan_split
```

The arguments can also be passed as `--config path_to_config.yaml`. Note that the configuration file overrides the arguments passed via command line.

#### Pretrained model

A baseline model trained using the parameters above can be downloaded from [here](http://www.robots.ox.ac.uk/~joon/data/baseline_unsuper.model).

You can check that the following script returns: `EER 11.8134`.

```
python ./trainSpeakerNet.py --eval --log_input True --save_path data/test --test_list /path/to/voxcelebs/test_list.txt --test_path /path/to/voxcelebs/voxceleb1 --initial_model baseline_unsuper.model 
```

#### Implemented loss functions
```
Prototypical (proto)
Angular Prototypical (angleproto)
```

#### Implemented models and encoders

Note that the model definitions are not compatible with those in [`voxceleb_trainer`](https://github.com/clovaai/voxceleb_trainer), since the spectrograms are extracted in the data loader.
```
ResNetSE34L (SAP)
```

#### Citation

Please cite the following if you make use of the code.

```
@inproceedings{huh2020augmentation,
  title={Augmentation adversarial training for unsupervised speaker recognition},
  author={Huh, Jaesung and Heo, Hee Soo and Kang, Jingu and Watanabe, Shinji and Chung, Joon Son},
  booktitle={Workshop on Self-Supervised Learning for Speech and Audio Processing, NeurIPS},
  year={2020}
}
```
