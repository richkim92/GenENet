# A simplified low‑channel EMG sensing for human‑machine interfaces
#### A modular framework for generative representation learning and downstream evaluation of EMG gesture prediction using low-channel wearble EMG device.
---

**[Kyun Kyu Kim](https://kyunkyukim.com)\#<sup>1</sup>, [Zhenan Bao](https://baogroup.stanford.edu)\*<sup>1</sup>**  
<sup>1</sup>Stanford University, CA, USA. Please refer manuscript for full author list. 

<!--  [![arXiv](https://img.shields.io/badge/arXiv%20paper-2504.11295-b31b1b.svg)](https://arxiv.org/abs/2504.11295)&nbsp;   -->

## Overview
We propose a Generative Electromyography Network (GenENet), a representation learning framework integrated with a wearable sensor system that leverages a simple, low‑channel‑count device to predict a broad spectrum of body kinematics traditionally reliant on high‑density EMG sensor arrays. This approach enables acquisition of information comparable to that obtained from high‑density EMG networks while using a more compact and wearable device with significantly fewer sensors, without compromising performance in human‑computer interaction tasks.
<!--![Teaser image](./assets/Cover_figure.png)-->

| <div align="center"><span style="font-size:12px;">Year</span></div> | <div align="center"><span style="font-size:12px;">2021<sup>[1]</sup></span></div> | <div align="center"><span style="font-size:12px;">2024<sup>[2]</sup></span></div> | <div align="center"><span style="font-size:12px;">2023<sup>[3]</sup></span></div> | <div align="center"><span style="font-size:12px;">2020<sup>[4]</sup></span></div> | <div align="center"><span style="font-size:12px;">2022<sup>[5]</sup></span></div> | <div align="center"><span style="font-size:12px;">**Ours**</span></div> |
|-------|-------------------------|------------|--------------------------|--------------------------------|------------------------|------|
| <div align="center"><span style="font-size:12px; font-weight:bold;">Outline</span></div> | <div align="center"><img src="assets/nat_elec_2021.png" width="200"/></div> | <div align="center"><img src="assets/tbme_2024.png" width="250"/></div> | <div align="center"><img src="assets/sci_report_2023.png" width="200"/></div> | <div align="center"><img src="assets/bio_eng_2020.png" width="50"/></div> | <div align="center"><img src="assets/IEEE_2022.png" width="70"/></div> | <div align="center"><img src="assets/ours_2025.png" width="400"/></div> |
| <div align="center"><span style="font-size:12px; font-weight:bold;">Channels</span></div> | <div align="center"><span style="font-size:12px;">64 EMG</span></div> | <div align="center"><span style="font-size:12px;">320 EMG</span></div> | <div align="center"><span style="font-size:12px;">64 EMG</span></div> | <div align="center"><span style="font-size:12px;">2 IMUs</span></div> | <div align="center"><span style="font-size:12px;">8 IMUs,<br>2 Cameras</span></div> | <div align="center"><span style="font-size:12px;">**6 EMG**</span></div> |
| <div align="center"><span style="font-size:12px; font-weight:bold;">Prediction</span></div> | <div align="center"><span style="font-size:12px;">13 gestures</span></div> | <div align="center"><span style="font-size:12px;">22 gestures</span></div> | <div align="center"><span style="font-size:12px;">65 gestures</span></div> | <div align="center"><span style="font-size:12px;">Gait dynamics</span></div> | <div align="center"><span style="font-size:12px;">Gait dynamics</span></div> | <div align="center"><span style="font-size:12px;">**26 gestures,<br> Gait dynamics**</span></div> |
| <div align="center"><span style="font-size:12px; font-weight:bold;">Area(cm²)</span></div> | <div align="center"><span style="font-size:12px;">240</span></div> | <div align="center"><span style="font-size:12px;">208</span></div> | <div align="center"><span style="font-size:12px;">>64</span></div> | <div align="center"><span style="font-size:12px;">>300</span></div> | <div align="center"><span style="font-size:12px;">>300</span></div> | <div align="center"><span style="font-size:12px;">**4.5**</span></div> |

<sub>
**References**  
[1] Nature Electronics, 2021  
[2] IEEE TBME, 2024  
[3] Scientific Reports, 2023  
[4] Front. Bioeng. Biotechnol, 2020  
[5] IEEE Transactions on Industrial Informatics, 2022
</sub>



##  Setup


### 1. Clone respository  

Clone this repository and navigate to the root directory.
```bash
git clone https://github.com/richkim92/GenENet.git
cd GenENet
```
---

### 2. Set up data

Sample dataset is stored in /dataset. Modify paths in `params.py` if needed.  
To download pretrained checkpoint `model.pt`, run:

```bash
python -m utils.download_model --model-name model
```
---

### 3. Pretraining

Run masked self-supervised pretraining. This will drive representation learning of entire 32 channels. 

```bash
python pretrain/train_pre.py
```
---

### 4. Downstream Classification

Run fine-tuning using pretrained encoder and downstream LSTM:

```bash
python downstream/train_post.py
```

By default, it loads the checkpoint from `./model_pt/model.pt`.

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
