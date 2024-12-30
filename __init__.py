import os
import torch
from ultralytics import YOLO

def get_gpu_index(target_gpu_name):
    """
    Searches for the GPU with the specified name and returns its index.

    Parameters:
        target_gpu_name (str): The name (or part of the name) of the target GPU.

    Returns:
        int: The index of the GPU if found.
        None: If the GPU is not found.
    """
    for i in range(torch.cuda.device_count()):
        current_gpu_name = torch.cuda.get_device_name(i)
        if target_gpu_name.lower() in current_gpu_name.lower():
            return i
    return None

def main():
    # Define the target GPU name
    TARGET_GPU_NAME = "RTX 4080"

    # Attempt to find the GPU index for the RTX 4080
    gpu_index = get_gpu_index(TARGET_GPU_NAME)

    if gpu_index is not None:
        device = f"cuda:{gpu_index}"
        print(f"Found {TARGET_GPU_NAME} at index {gpu_index}. Using device: {device}")
    else:
        # If RTX 4080 is not found, decide whether to fallback or raise an error
        print(f"{TARGET_GPU_NAME} not found. Falling back to CPU or another GPU if available.")
        
        # Option 1: Use the first available GPU
        if torch.cuda.is_available():
            device = "cuda:0"
            print(f"Using first available GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("No GPU available. Using CPU.")

    # Optional: Display detailed GPU information
    if "cuda" in device:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        if gpu_index is not None:
            print(f"Using GPU: {torch.cuda.get_device_name(gpu_index)}")
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Initialize the YOLOv8 model
    # You can choose different model sizes: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    # Here, we're using yolov8s for a balance between speed and accuracy
    model = YOLO("yolov8s.yaml")  # Alternatively, use "yolov8s.pt" to fine-tune from a pretrained checkpoint

    # Start training
    model.train(
        data=os.path.join("C:/Users/zahit/OneDrive/Masaüstü/GitHub/BMR-Power-Outlet-Recognition/data", "data.yaml"),          # Path to your dataset config
        epochs=100,                                      # Number of training epochs
        batch=16,                                        # Batch size (adjust based on GPU memory)
        imgsz=640,                                       # Image size (YOLOv8 typically uses square images)
        optimizer="AdamW",                               # Optimizer choice
        lr0=1e-4,                                        # Initial learning rate
        lrf=0.01,                                        # Final learning rate factor for scheduler
        patience=20,                                     # Early stopping patience
        device=device,                                   # Device selection
        project=os.path.join("runs", "power_socket"),    # Output directory for training logs and checkpoints
        name="yolov8s_powersocket",                      # Experiment name
        cache=True,                                      # Cache images for faster training
        workers=8,                                       # Number of data loader workers
        cos_lr=True,                                     # Use cosine learning rate scheduler
        seed=42,                                         # Seed for reproducibility
        amp=True,                                        # Use Automatic Mixed Precision for faster training and lower memory usage
        augment=True,                                    # Enable data augmentations
    )

    # Evaluate the model on the validation set after training
    print("Training complete. Evaluating on the validation set...")
    metrics = model.val()
    print(f"Validation results: {metrics}")

    # Optional: Save the trained model
    # model.save(os.path.join("runs", "power_socket", "best_model.pt"))

    # Optional: Perform inference on new images
    # predictions = model.predict(source="path/to/test/images", conf=0.25)
    # print(predictions)

if __name__ == "__main__":
    main()
