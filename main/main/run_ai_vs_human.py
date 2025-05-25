import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import os

# Categories used during training
def load_categories(file_path="../categories.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

categories = load_categories()

# Load the trained model
model = tf.keras.models.load_model("doodle_model.h5")

# Create results CSV file
csv_filename = "ai_vs_human_results.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, mode="a", newline="") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Image #", "Correct Label", "Human Guess", "AI Guess", "AI Correct?"])

    image_num = 1

    for category_index, category_name in enumerate(categories):
        data = np.load(f"npy_data/{category_name}.npy")
        samples = data[:5]  # Test 5 images per category

        for image in samples:
            # Prepare image
            image_reshaped = image.reshape(28, 28)
            image_input = image_reshaped.reshape(1, 28, 28, 1) / 255.0

            # Display to human
            plt.imshow(image_reshaped, cmap="gray")
            plt.title(f"Image #{image_num}")
            plt.axis("off")
            plt.show()

            # Print options
            print(f"🤔 What do YOU think this is?")
            for i, cat in enumerate(categories):
                print(f"{i}: {cat}")
            
            # Get and validate input
            while True:
                try:
                    human_index = int(input("Enter the number of your guess: "))
                    if 0 <= human_index < len(categories):
                        break
                    else:
                        print("❌ Invalid number. Try again.")
                except ValueError:
                    print("❌ Please enter a valid number.")

            human_guess = categories[human_index]

            # AI prediction
            prediction = model.predict(image_input, verbose=0)
            predicted_index = np.argmax(prediction)
            ai_guess = categories[predicted_index]

            # Record result
            correct_label = category_name
            ai_correct = ai_guess == correct_label

            print(f"✅ Correct label: {correct_label}")
            print(f"🧠 AI guessed: {ai_guess}")
            print(f"🎯 {'AI was right!' if ai_correct else 'AI was wrong!'}\n")

            # Write to CSV
            writer.writerow([
                image_num,
                correct_label,
                human_guess,
                ai_guess,
                "Yes" if ai_correct else "No"
            ])

            image_num += 1

print(f"\n📄 All results saved to {csv_filename}")
