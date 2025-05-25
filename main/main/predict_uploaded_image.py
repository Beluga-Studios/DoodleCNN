import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

# === CONFIG ===
MODEL_PATH = "doodle_model.h5"
IMAGE_SIZE = (28, 28)

# === Categories (must match training order and count) ===
def load_categories(file_path="../categories.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

categories = load_categories()

# === Load model once ===
model = tf.keras.models.load_model(MODEL_PATH)

print("🎨 AI Doodle Guesser")
print("Type a file path, folder path (with .npy files), or 'exit' to quit.")

while True:
    path = input("\n📁 Enter image file or dataset folder: ").strip()
    if path.lower() == "exit":
        print("👋 Goodbye!")
        break

    true_label = None  # Track real label if using dataset

    if os.path.isdir(path):
        npy_files = [f for f in os.listdir(path) if f.endswith(".npy")]
        if not npy_files:
            print("❌ No .npy files found in that folder.")
            continue
        chosen_file = random.choice(npy_files)
        full_path = os.path.join(path, chosen_file)
        category_name = chosen_file.replace(".npy", "")
        true_label = category_name

        # Load random image from .npy
        try:
            data = np.load(full_path)
        except Exception as e:
            print(f"❌ Failed to load {full_path}:", e)
            continue

        index = random.randint(0, len(data) - 1)
        img_array = data[index].reshape(28, 28) / 255.0
        print(f"🎲 Random image from '{category_name}' (index {index})")

    elif os.path.isfile(path):
        try:
            img = Image.open(path).convert("L").resize(IMAGE_SIZE)
            img_array = np.array(img) / 255.0
        except Exception as e:
            print(f"❌ Failed to open image:", e)
            continue

    else:
        print("❌ Invalid path.")
        continue

    # === Predict ===
    img_input = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_input)[0]
    top_indices = prediction.argsort()[-3:][::-1]  # Top 3 guesses

    # === Show image ===
    plt.imshow(img_array, cmap="gray")
    plt.axis("off")
    plt.title("What is this?")
    plt.show()

    # === Show predictions ===
    print("\n🤖 AI guesses:")
    ai_top_guess = categories[top_indices[0]] if top_indices[0] < len(categories) else "Unknown"
    for rank, idx in enumerate(top_indices, 1):
        if idx < len(categories):
            confidence = prediction[idx] * 100
            print(f"  {rank}. {categories[idx]} ({confidence:.2f}% confident)")
        else:
            print(f"  {rank}. [Unknown category index: {idx}]")

    # === Compare to true label ===
    if true_label:
        print(f"\n✅ Correct label: {true_label}")
        if ai_top_guess == true_label:
            print("🎉 AI was CORRECT!")
        else:
            print("❌ AI was WRONG.")
