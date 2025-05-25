import matplotlib.pyplot as plt
import numpy as np

# Make a simple fake image
image = np.random.rand(28, 28)

plt.imshow(image, cmap="gray")
plt.title("Test Image")
plt.axis("off")
plt.show(block=False)
plt.pause(0.1)
input("Do you see the image? Press Enter to close it...")
plt.close()
print("✅ It worked!")
