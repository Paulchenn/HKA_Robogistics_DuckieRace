from ultralytics import YOLO
import os

# Check if the current working directory is the same as the script's directory
if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    # If not, change the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load a model
model = YOLO("yolov8n.pt")  # load an official model, here nano (smallest but fastest model)

# Train the model
model.train(
    data=os.path.join(os.getcwd(), "data/data.yaml"),
    epochs=60,
    imgsz=640,
    project="results",
    name="duckBotFreeOcc",
    patience=20,
    batch=16,
    device="cpu",  # Use GPU 0, change to "cpu" if you want to train on CPU
    save_period=0,  # Save model every epoch
    save=True,  # Save the model after training
    exist_ok=True,  # Overwrite existing results folder
    warmup_epochs=3,  # Number of warmup epochs
)