# DoodleCNN
doodle recognation code that uses the Quick, Draw! Dataset to geuss doodles.
its for a [science fair](https://docs.google.com/document/d/19foik4s-7mHe20O_G7mDw6pEZspBnqWRPBzb7SAfpuk)

This is a human vs AI guessing game based on hand-drawn doodles from the [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset). It tests whether humans or AI are better at recognizing simple sketches.

**Note:** This only works on Mac

---

## 📦 Features

- Trains a Convolutional Neural Network (CNN) on selected categories.
- Saves the trained model for reuse.
- Includes a human test mode to guess drawings and compare against the AI.
- Scripts to download both raster and vector-based QuickDraw data (`.npy` and `.ndjson`).
- Includes a Godot app for human testing on the web or Chromebook.

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/beluga-studios/doodle-cnn.git
cd doodle-recognition-ai/main/
```

### 2. Install Python Dependencies

Make sure you have Python 3.9+ and `pip`. Then run:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, use:

```bash
pip install tensorflow numpy matplotlib ndjson
```

If you're using `ndjson` vector files:

```bash
pip install ndjson
```

### 4. Test with MNIST dataset

Run this and in about 5 minutes you should see results if it worked:

```bash
python tests/MNIST_cnn.py
```

### 4. Download Data

First go to [categories.txt](main/categories.txt) and delete all the categories you dont want. we suggest only keeping a few unless you want to wait 15 hours. **DO NOT EDIT THE FILE AFTER YOU HAVE STARTED RUNNING THE SCRIPTS**

You can download `.npy` (bitmap, for the AI) or `.ndjson` (vector, for the Godot human test) data files from the [QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset#download-the-dataset). To download selected categories

For the AI:

```bash
python main/download_npy_categories.py
```
For the Godot human test:

```bash
python human-test/download_vectors.py.py 10 # Change 10 to how much images you want from each category
```

---

## 🤖 Training the AI

To train the model:

```bash
python main/doodle_cnn.py
```

- The model is saved as `doodle_model.h5`.

---

## 🧠 Predict Using Trained Model

To test an image from the dataset and compare against AI:

```bash
python main/predict_uploaded_image.py
```

- Loads a 28x28 grayscale image and shows AI's top guesses.

---

## 🕹️ Human Testing in Godot

For Chromebook or web testing, open the `human-test/` folder in [Godot](https://godotengine.org) (v4.x).

### Godot App Setup

- If you havent already, run this:

```bash
python human-test/download_vectors.py.py 10 # Change 10 to how much images you want from each category
```

and make sure `.ndjson` files are inside `res://images/`.

- Run the project (`⌘+B` on Mac) to begin testing.
- Shows a vector image, lets user guess, and tracks accuracy.
- Write down what accuracy they got.

---

## 📜 License

MIT License. Feel free to use and modify for your experiments or projects!

---

## ✍️ Made by

**Beluga Studios** — experimental games and AI tools
