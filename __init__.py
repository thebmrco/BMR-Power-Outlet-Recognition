import os
import torch
import torchvision
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
    target_gpu_name = target_gpu_name.lower()
    for i in range(torch.cuda.device_count()):
        current_gpu_name = torch.cuda.get_device_name(i).lower()
        if target_gpu_name in current_gpu_name:
            return i
    return None


def main():
    # ----------------------------------------------------------------------
    # Attempt to find the target GPU by name
    # ----------------------------------------------------------------------
    TARGET_GPU_NAME = "NVIDIA"  # Change to match the target GPU name

    gpu_index = None
    if torch.cuda.is_available():
        # First try to find the GPU by name
        gpu_index = get_gpu_index(TARGET_GPU_NAME)

        if gpu_index is not None:
            device = f"cuda:{gpu_index}"
            print(f"Found target GPU '{TARGET_GPU_NAME}' at index {gpu_index}.")
        else:
            # If we don't find the target by name, just use the first available GPU
            device = "cuda:0"
            print(
                f"Target GPU '{TARGET_GPU_NAME}' not found. Using first available GPU: "
                f"{torch.cuda.get_device_name(0)}."
            )
    else:
        # Fallback to CPU if no GPU is available
        device = "cpu"
        print("No GPU available. Using CPU.")

    # ----------------------------------------------------------------------
    # Optional: Display detailed GPU information
    # ----------------------------------------------------------------------
    if device != "cpu":
        print(f"Using device: {device}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(int(device.split(':')[-1]))}")
    else:
        print("Using CPU.")

    # ----------------------------------------------------------------------
    # Check if TorchVision supports GPU-based NMS
    # ----------------------------------------------------------------------
    try:
        _ = torchvision.ops.nms
    except NotImplementedError:
        print("TorchVision doesn't support GPU-based NMS. Falling back to CPU.")
        device = "cpu"

    # ----------------------------------------------------------------------
    # Initialize the YOLOv8 model
    # ----------------------------------------------------------------------
    # You can choose different model sizes: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    # Here, we're using yolov8s for a balance between speed and accuracy
    model = YOLO(
        "yolov8s.yaml"
    )  # Or "yolov8s.pt" for fine-tuning a pretrained checkpoint

    # ----------------------------------------------------------------------
    # Start training
    # ----------------------------------------------------------------------
    model.train(
        data=os.path.join("datasets", "data.yaml"),  # Relative path to dataset
        epochs=50,  # Number of training epochs
        batch=8,  # Reduced batch size
        imgsz=640,  # Image size
        optimizer="AdamW",  # Choice of optimizer
        lr0=1e-4,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor (scheduler)
        patience=20,  # Early stopping patience
        device=device,  # Device selection
        project=os.path.join("runs", "power_socket"),  # Output directory
        name="yolov8s_powersocket",  # Experiment name
        cache="disk",  # Use disk caching instead of RAM
        workers=8,  # Number of data loader workers
        cos_lr=True,  # Use cosine LR scheduler
        seed=42,  # Seed for reproducibility
        amp=True,  # Automatic Mixed Precision
        augment=True,  # Enable data augmentations
    )

    # ----------------------------------------------------------------------
    # Evaluate the model on the validation set after training
    # ----------------------------------------------------------------------
    print("Training complete. Evaluating on the validation set...")
    metrics = model.val()
    print(f"Validation results: {metrics}")

    # ----------------------------------------------------------------------
    # Optional: Save the trained model
    # ----------------------------------------------------------------------
    # model.save(os.path.join("runs", "power_socket", "best_model.pt"))

    # ----------------------------------------------------------------------
    # Optional: Perform inference on new images
    # ----------------------------------------------------------------------
    # predictions = model.predict(source="path/to/test/images", conf=0.25)
    # print(predictions)


if __name__ == "__main__":
    main()
