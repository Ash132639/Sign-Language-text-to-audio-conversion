import pickle
import matplotlib.pyplot as plt

# Load the history data
with open("sequential_model_score.pkl", "rb") as f:
    history = pickle.load(f)

assert all(key in history for key in ["loss", "val_loss", "accuracy", "val_accuracy", "time"]), "Missing keys in history data"

# Plotting the Loss (Train and Validation)
plt.figure(figsize=(10, 6))
plt.plot(history["loss"], label="Train Loss", color="blue")
plt.plot(history["val_loss"], label="Validation Loss", color="red")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("model_loss.png")  # Save the figure
plt.show()

# Plotting the Accuracy (Train and Validation)
plt.figure(figsize=(10, 6))
plt.plot(history["accuracy"], label="Train Accuracy", color="blue")
plt.plot(history["val_accuracy"], label="Validation Accuracy", color="red")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("model_accuracy.png")  # Save the figure
plt.show()

# Plotting the Epoch Times
plt.figure(figsize=(10, 6))
plt.plot(history["time"], label="Epoch Time", color="green")
plt.title("Epoch Times")
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.savefig("epoch_times.png")  # Save the figure
plt.show()
