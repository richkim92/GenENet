# GenENet
#### A modular framework for self-supervised pretraining and downstream evaluation of EMG gesture prediction using GenENet.
---

**[Kyun Kyu Kim](https://kyunkyukim.com)\#<sup>1</sup>, [Zhenan Bao](https://baogroup.stanford.edu)\*<sup>1</sup>**  
<sup>1</sup>Stanford University, CA, USA. Please refer manuscript for full author list. 

<!--  [![arXiv](https://img.shields.io/badge/arXiv%20paper-2504.11295-b31b1b.svg)](https://arxiv.org/abs/2504.11295)&nbsp;   -->

## Overview
<i>We propose a Generative Electromyography Network (GenENet), a generative algorithm combined with wearable sensor that extrapolates limited sensor counts to reconstruct muscle activity in unseen regions. This approach allows for gathering information equivalent to those from high-density EMG sensor networks, but using a more compact, wearable device with much reduced sensor counts, without sacrificing performance</i>
![Teaser image](./assets/Cover_figure.png)

## Dependencies
The requirements for this code are the same as [MAE](https://arxiv.org/abs/2111.06377).


## 📁 Project Structure

```
GenENet/
├── models/                  # Core model components 
├── pretrain/                # Pretraining phase
│   ├── dataset/             # 32ch Datasets for pretraining
│   ├── model_pt/            # Saved pretraining checkpoints
│   ├── params.py            
│   └── train_pre.py         # Pretraining script
├── utils                    # Shared utilities (augmentation, scheduler, dataset utils)
└── downstream.py            # Downstream fine-tuning and evaluation
    ├── dataset/             # Dataset consisted with sample Sign-language labels
    ├── model_pt/            
    ├── params.py
    └── train_post.py        # Posttraining script
```
---

##  Setup

### 1. Set up data

Organize your dataset in the required format. Modify paths in `params.py` if needed.  
Please download pretrained checkpoint `model.pt` from the following link:  
https://drive.google.com/drive/folders/1LGjDGhr8GPF6FJcVxDECCW5eh7b4y-Zg?usp=drive_link

---

### 2. Pretraining

Run masked autoencoding pretraining. This will drive representation learning of entire 32 channels. 

```bash
python pretrain/train_pre.py
```
---

### 3. Downstream Classification

Run fine-tuning using pretrained encoder and downstream LSTM:

```bash
python train_post.py
```

By default, it loads the checkpoint from `./pretrain/model_pt/model.pt`.

---

## Results

| Task                  | Accuracy |
|-----------------------|----------|
| EMG Sign Language Translation   | 93.6 %    |
| EMG Gait Force Prediction | 6.21 % (Relative RSME)    |

---

## Author

- Kyun Kyu (Richard) Kim ([@richkim92](https://github.com/richkim92))
- Contact: enthusiakk@gmail.com
- Affiliation: Stanford University

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
