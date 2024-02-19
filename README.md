See the attached PDF and video for detailed explnation of the project.

The code for the base-net [Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/pdf/1811.00751.pdf) is taken from https://github.com/Pay20Y/SAR_TF

Some edits were made to fit the base network to handling new languages. The code for performing meta learning is inside meta_learn.py. The code for regular training is in training.py.

The pretrained backbone (feature extractor) of the network is necessary for running either regular or meta training, and is not included here due to it's size. It's available at:
https://drive.google.com/file/d/1JSCCXYlpCiSuLj41jYhPQDPU5PGVo3UF/view?usp=sharing
Note that this was uploaded by a third party (not us / the paper's authors)

The datasets we used are also not included due to their size.