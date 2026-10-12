# Tomato Leaf Disease Detection

An image classifier that looks at a photo of a tomato leaf and says whether it is healthy or which of 9 diseases it shows, with a Streamlit app for demonstration.

## Problem and scope

Farmers and farm advisors need a quick first check of leaf diseases, since manual inspection is slow and needs an expert. This project builds a model that gives that first check.

- **In scope:** tomato leaves only, 10 classes (9 diseases + healthy), a baseline CNN, transfer learning with MobileNetV2 and ResNet50, evaluation on a held-out test set, and a Streamlit app
- **Out of scope:** real diagnosis without an expert, other crops, field photos, a mobile app
- **Why tomato only:** with one crop, the model has to learn disease features instead of telling crops apart

## Dataset

- PlantVillage, loaded with TensorFlow Datasets: https://www.tensorflow.org/datasets/catalog/plant_village
- Filtered to the 10 tomato classes: 18,159 images
- Split 80/10/10 per class with seed 42: train 14,522, val 1,812, test 1,825

## Results (test set)

| Model | Accuracy | Macro-F1 | Top-3 accuracy |
|---|---|---|---|
| Baseline CNN | 90.2% | 0.867 | 98.5% |
| MobileNetV2 frozen | 91.5% | 0.882 | 99.0% |
| **MobileNetV2 fine-tuned (used in the app)** | **96.8%** | **0.952** | **99.7%** |
| ResNet50 frozen | 96.7% | 0.958 | 99.7% |

MobileNetV2 fine-tuned is used in the app: about as accurate as ResNet50, 10 times smaller (8.7 MB vs 90 MB), and fewer diseased leaves called healthy (7 vs 13).

## Project structure

```
plant-disease-detection/
├── notebooks/
│   ├── 00_data_access.ipynb       load PlantVillage, filter to tomato, split, data checks
│   ├── 01_eda.ipynb               class balance, sample images, image checks
│   ├── 02_baseline_cnn.ipynb      baseline CNN trained from scratch
│   ├── 03_transfer_learning.ipynb MobileNetV2 and ResNet50, fine-tuning, experiment log
│   ├── 04_evaluation.ipynb        test results, confusion matrices, errors, latency
│   └── 05_demo_samples.ipynb      demo images for the app, warning level
├── app/
│   ├── app.py                     Streamlit app
│   └── samples/                   test images used in the demo
├── models/
│   └── mobilenet_finetuned.keras  model used by the app (the only model in the repo)
├── reports/figures/               charts saved by the notebooks
├── check_tf.py                    checks TensorFlow and the GPU
└── requirements.txt               exact package versions
```

The `data/` folder and the other trained models are not in the repo because they are large. The notebooks create them.

## Setup

Built and tested on a MacBook (M4 Pro), Python 3.12, TensorFlow 2.19.1.

```bash
git clone https://github.com/gyanesh5522apte-web/plant-disease-detection.git
cd plant-disease-detection
conda create -n plant python=3.12
conda activate plant
pip install -r requirements.txt
python check_tf.py
```

- `check_tf.py` prints the TensorFlow version and whether a GPU was found
- **Not on a Mac?** Remove the `tensorflow-metal` line from `requirements.txt` before installing; it only works on Apple chips

## Running the notebooks

Run them in order, 00 to 05.

- Each notebook starts with `os.chdir("/Users/eloelo/Projects/plant-disease-detection")`. Change it to where you put the project
- In notebook 00, also change `data_dir` and `splits_dir`
- Notebook 00 downloads about 830 MB the first time and saves the splits to `data/splits_tfds`
- Notebooks 02 and 03 train the models, which takes about an hour in total on an M4 Pro

You don't need to run the notebooks to use the app. The model it needs is already in `models/`.

## Running the app

From the project folder:

```bash
conda activate plant
streamlit run app/app.py
```

The app opens in the browser at http://localhost:8501. Always start it from the project folder, since the model and sample paths are relative to it. To stop it, press Ctrl + C in the terminal.

## How to use the app

1. Choose **Upload a photo** and pick a photo of one tomato leaf, or choose **Try a sample** and pick one from the list
2. The result shows the most likely class, the confidence, and the next two possibilities
3. Read **What this means** and **Suggested next step** for a plain explanation
4. If confidence is below 70%, a yellow warning says the model is unsure
5. Open **How it works**, **Limitations** and **About the model** at the bottom for details

The samples are test images the model never saw in training. "Late blight (known failure case)" is a leaf the model gets wrong: it says healthy, but with a warning.

## Limitations

- Tomato leaves only; any other image is still forced into one of the 10 classes
- Trained and tested only on lab photos (one leaf, plain background). Field photos were not formally evaluated, and the model can be wrong on them, even with high confidence
- Weakest on Early blight (78% found in testing), often confused with Target spot
- Called 7 of 1,665 diseased test leaves healthy
- For advice and learning only, not a diagnosis. Always confirm with an agricultural expert
