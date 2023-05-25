import torch
import dill
import os

MODEL_DIR = "./models"

# Load the model and metadata
model_filename = os.path.join(MODEL_DIR, "g_best_model.pth")
with open(model_filename, "rb") as f :
  metadata = dill.load(f)

# Access the metadata
epoch = metadata["epoch"]
batch_iteration = metadata["batch_iteration"]
g_loss = metadata["g_loss"]

print(f"{model_filename} model of epoch {epoch}, batch iteration {batch_iteration}, g_loss {g_loss}")

# Load the model and metadata
model_filename = os.path.join(MODEL_DIR, "g_best_epoch_model.pth")
with open(model_filename, "rb") as f :
  metadata = dill.load(f)

# Access the metadata
epoch = metadata["epoch"]
batch_iteration = metadata["batch_iteration"]
g_loss = metadata["g_loss"]

print(f"{model_filename} model of epoch {epoch}, batch iteration {batch_iteration}, g_loss {g_loss}")