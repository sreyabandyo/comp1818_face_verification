# COMP1818 - AI Facial Recognition and Biometric Authentication

## 1. Project Overview
This project investigates the performance and robustness of a pretrained AdaFace IR18 deep learning model for face verification.
The experiments examine how variations in facial pose, age, lighting conditions and image quality affect verification accuracy. 

## 2. Objectives
- Evaluate AdaFace IR18 across six face verification benchmark datasets.
- Investigate the effects of simulated brightness changes.
- Examine how Gaussian blur affects face verification accuracy.
- Compare AdaFace IR18 against a majority-class baseline.

## 3. Model and Technologies

### 3.1 Model: AdaFace IR18 (pretrained on CASIA-WebFace)

### 3.2 Technologies:
- Python
- PyTorch
- OpenCV
- NumPy and Pandas
- Sckit-learn
- Matplotlib
- Google Colab

### 3.3 Pretrained Weights
The pretrained AdaFace IR18 checkpoint (trained on CASIA-WebFace) was downloaded from the "Pretrained Models" table (R18, CASIA-WebFace) in the official AdaFace repository (Kim et al., 2022): https://github.com/mk-minchul/AdaFace

Checkpoint download: https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view?usp=sharing

No training or fine-tuning was performed. The pretrained model was used for inference only. Note that this model expects BGR input rather than RGB.

## 4. Benchmark Datasets
The model was evaluated using six face verification benchmark datasets:

| Dataset | Verification Accuracy |
|---------|-----------------------|
| LFW | 99.23 |
| CFP-FF | 99.10 |
| CFP-FP | 91.90 |
| AgeDB-30 | 91.95 |
| CALFW | 92.60 |
| CPLFW | 86.07 |

Accuracy was measured using 10-fold cross-validation.

### 4.1 Dataset Sources
The six verification benchmarks (LFW, CFP-FF, CFP-FP, AgeDB-30, CALFW and CPLFW) were loaded as pre-aligned 112x112 `.bin` files from the `faces_webface_112x112` archive. This is the InsightFace-provided dataset package (CASIA-WebFace, preprocessed to 112x112) that the official AdaFace repository's data preparation guide (`README_TRAIN.md`) directs users to: https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_

Original benchmark papers:
- LFW: Huang et al. (2007)
- CFP-FF / CFP-FP: Sengupta et al. (2016)
- AgeDB-30: Moschoglou et al. (2017)
- CALFW: Zheng, Deng and Hu (2017)
- CPLFW: Zheng and Deng (2018)

The dataset files are not included in this repository because of their size. To reproduce the results, download the archive from the link above (CASIA-WebFace entry) and extract it to `/content/data/faces_webface_112x112/` in Google Colab (or update the path in the notebook).

## 5. Robustness Experiments

### 5.1 Lighting Conditions
Five brightness conditions were evaluated, ranging from very dark to very bright.

Verification accuracy ranged from 98.62% to 99.23%, indicating relatively limited sensitivity to the simulated brightness changes.

### 5.2 Image Quality
Gaussian blur was applied at increasing levels of intensity.

Verification accuracy decreased from 99.23% on the original LFW benchmark to 92.53% under very strong blur, indicating that reduced image sharpness can affect verification reliability.

### 5.3 Baseline Comparison
A majority-class classifier was used as a simple baseline.

On the same 1,200 held-out LFW pairs

| Method | Test Accuracy |
|--------|---------------|
| Majority-class baseline | 49.58% |
| AdaFace IR18 | 98.58% |

## 6. Repository Contents
- comp1818_face_verification.ipynb - implementation, experiments, visualisations and evaluation results.
- results_pretrained_ir18.csv - verification results across the six benchmark datasets.
- README.md - project documentation.

## 7. Limitations
The lighting and blur experiments use artificially modified images rather than naturally occurring environmental conditions. Results are specific to the pretrained AdaFace IR18 model and the datasets evaluated.

## 8. Conclusion
AdaFace IR18 demonstrated strong face verification performance under standard benchmark conditions. However, accuracy decreased on more challenging datasets and under severe Gaussian blur, highlighting the importance of image quality and facial variation in biometric authentication systems.

## 9. Acknowledgements
The project uses the pretrained AdaFace model developed by Kim et al. (2022) 


Original AdaFace repository: [AdaFace GitHub](https://github.com/mk-minchul/AdaFace)
