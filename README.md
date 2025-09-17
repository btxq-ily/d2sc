# D2SC: DUAL-STAGE DIFFUSION-GUIDED SEMANTIC-VISUAL COUPLING FOR ZERO-SHOT LEARNING #

Generative zero-shot learning remains challenging in practice. When conditioning on diffusion timesteps, semantics become step-dependent: the same concept drifts across noise levels, weakening conditional guidance for synthesis. Adversarial alignment alone further leaves a mismatch between synthesized and real features, both in global statistics and class-wise geometry; and in multi-branch setups, discriminator heads can produce inconsistent gradients that destabilize training. To address these issues, we design D2SC (Dual-Stage Diffusion-Guided Semantic-Visual Coupling) with a simple rationale: first stabilize the conditional prior, then align the visual space. Concretely, the Contrastive Prior Regularizer (CPR) enforces timestep consistency to suppress conditional drift, while the Visual Feature Synthesizer (VFS) performs multi-alignment via maximum mean discrepancy and center loss, complemented by lightweight discriminator distillation for coherent feedback. The two modules share semantic mappings and a unified schedule. With a single hyper-parameter setting across datasets, D2SC achieves robust zero-shot learning (ZSL) and generalized zero-shot learning (GZSL) gains.

## Prerequisites
+ Python 3.8
+ Pytorch 1.12.1
+ torchvision 0.13.1
+ scikit-learn 1.3.0
+ scipy=1.10.0
+ numpy 1.24.3
+ numpy-base 1.24.3
+ pillow 9.4.0

## Data preparation

We trained the model on three popular ZSL benchmarks: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html) and [AWA2](http://cvml.ist.ac.at/AwA2/) following the data split of [xlsa17](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip).

Our fine-tuned features can be downloaded from https://drive.google.com/drive/folders/1MW_GPqN7g9idJrtYg8eLszqFMXj-YKrq?usp=drive_link.

The data percent splitting can be downloaded from https://drive.google.com/drive/folders/1erHqyL42wJ1b7oWKkrPTyb2EoCxkFVqb?usp=drive_link. 

### Download Dataset 

Firstly, download these datasets as well as the xlsa17 and our data splitting and fine-tuned features. Then decompress and organize them as follows: 
```
.
├── Dataset
│   ├── CUB/CUB_200_2011/...
│   ├── SUN/images/...
│   ├── AWA2/Animals_with_Attributes2/...
│   ├── Attribute/w2v/...
│   └── xlsa17/data
│              ├── AWA2
│                  ├── att_splits.mat
│                  ├── ce_ce.mat
│                  ├── con_paco.mat
│                  ├── split_10percent.mat
│                  ├── split_30percent.mat
│                  └── res101.mat
│              ├── CUB/...
│              └── SUN/...
└── ···
```

## Training

To train and evaluate ZSL and GZSL models, please run the file `./scripts/train_awa2_d2sc_DRG.py` then the scripts `./scripts/train_awa2_d2sc_DFG`, e.g.:
```
python ./scripts/train_awa2_zerodiff_DRG.py
```
Then
```
python ./scripts/train_awa2_zerodiff_DFG.py
```



## Customized Fine-tuning
We also provide our fine-tuning code in  `./FineTune`. You can change it with your own loss or algorithm.

## Bibtex ##
If this work is helpful for you, please cite our paper

## References
Parts of our codes based on:
* [akshitac8/tfvaegan](https://github.com/akshitac8/tfvaegan)
* [ZhishengXiao/DenoisingDiffusionGAN](https://github.com/NVlabs/denoising-diffusion-gan)
* [Jiequan/PACO](https://github.com/dvlab-research/Parametric-Contrastive-Learning)
* [ZERODIFF](https://github.com/FouriYe/ZeroDiff_ICLR25/tree/main/datasets)
