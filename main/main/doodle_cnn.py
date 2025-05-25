import os
import urllib.request
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Define categories
def load_categories(file_path="../categories.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

categories = load_categories()

# Step 2: Create a folder to store data
os.makedirs("npy_data", exist_ok=True)

# Step 3: Download .npy files using urllib (works on all platforms)
for category in categories:
    filename_url = category.replace(" ", "%20") + ".npy"
    filename_local = f"npy_data/{category}.npy"
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{filename_url}"

    if not os.path.exists(filename_local):
        print(f"📥 Downloading {category}...")
        urllib.request.urlretrieve(url, filename_local)
        print(f"✅ Saved to {filename_local}")

# Step 4: Load and label data
images = []
labels = []

for i, category in enumerate(categories):
    data = np.load(f"npy_data/{category}.npy")
    try:
        data = np.load(f"npy_data/{category}.npy")
        if data.shape[1] != 784:
            raise ValueError("Wrong shape")
    except Exception as e:
        print(f"❌ Skipping {category}: {e}")
        continue
    images.append(data[:3000])  # Load 3000 samples per category
    labels += [i] * 3000

X = np.concatenate(images)
y = np.array(labels)

# Step 5: Preprocess data
X = X / 255.0  # Normalize pixel values
X = X.reshape(-1, 28, 28, 1)  # Reshape for CNN
y = tf.keras.utils.to_categorical(y, num_classes=len(categories))  # One-hot encode labels

# Step 6: Shuffle and split (80% train, 20% test)
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 7: Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(categories), activation="softmax")
])

# Step 8: Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Step 9: Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Step 10: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")

# Step 11: Save the model for later use
model.save("doodle_model.h5")
print("💾 Model saved as doodle_model.h5")

# Step 12: Plot accuracy
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
