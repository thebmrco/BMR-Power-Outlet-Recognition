import os
import torch
import torchvision
from ultralytics import YOLO
from pathlib import Path  # For cleaner path handling


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
    if not torch.cuda.is_available():
        return None
    for i in range(torch.cuda.device_count()):
        current_gpu_name = torch.cuda.get_device_name(i).lower()
        if target_gpu_name in current_gpu_name:
            return i
    return None


def main():
    # ----------------------------------------------------------------------
    # GPU Configuration
    # ----------------------------------------------------------------------
    TARGET_GPU_NAME = (
        "NVIDIA"  # Change if you have a specific preference like "RTX 3090" or "A100"
    )
    # "NVIDIA" will match any NVIDIA GPU.

    gpu_index = None
    if torch.cuda.is_available():
        gpu_index = get_gpu_index(TARGET_GPU_NAME)

        if gpu_index is not None:
            device = f"cuda:{gpu_index}"
            print(
                f"Found target GPU '{TARGET_GPU_NAME}' (or compatible) at index {gpu_index}: {torch.cuda.get_device_name(gpu_index)}."
            )
        else:
            if torch.cuda.device_count() > 0:
                device = "cuda:0"
                print(
                    f"Target GPU '{TARGET_GPU_NAME}' not specifically found. "
                    f"Using first available GPU: {torch.cuda.get_device_name(0)} (cuda:0)."
                )
            else:
                device = "cpu"
                print("CUDA is_available but no devices found. Using CPU.")
    else:
        device = "cpu"
        print("No CUDA-compatible GPU available. Using CPU.")

    # ----------------------------------------------------------------------
    # Display detailed device information
    # ----------------------------------------------------------------------
    if device != "cpu":
        device_id_for_info = int(device.split(":")[-1]) if ":" in device else 0
        print(f"Using device: {device}")
        print(f"Selected GPU Name: {torch.cuda.get_device_name(device_id_for_info)}")
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"Total Number of GPUs available: {torch.cuda.device_count()}")
    else:
        print("Using CPU for training.")

    # ----------------------------------------------------------------------
    # Check TorchVision NMS support
    # ----------------------------------------------------------------------
    try:
        _ = torchvision.ops.nms
        print("TorchVision GPU-based NMS is available.")
    except (NotImplementedError, AttributeError):
        print("TorchVision doesn't support GPU-based NMS or it's not found.")

    # ----------------------------------------------------------------------
    # Initialize the YOLOv8 model
    # ----------------------------------------------------------------------
    model_name = "yolov8s.pt"
    print(f"Initializing YOLO model with '{model_name}' for fine-tuning.")
    model = YOLO(model_name)

    # ----------------------------------------------------------------------
    # Dataset Configuration
    # ----------------------------------------------------------------------
    dataset_yaml_path = Path("datasets") / "data.yaml"
    print(f"Using dataset configuration: {dataset_yaml_path.resolve()}")
    if not dataset_yaml_path.exists():
        print(f"ERROR: Dataset YAML file not found at {dataset_yaml_path.resolve()}")
        print("Please ensure the path is correct and the file exists.")
        return

    # ----------------------------------------------------------------------
    # Training Parameters
    # ----------------------------------------------------------------------
    epochs = 50
    batch_size = 8
    image_size = 640
    optimizer_choice = "AdamW"
    initial_lr = 1e-4
    final_lr_factor = 0.01
    early_stopping_patience = 20

    project_dir = Path("runs") / "power_socket_detection"
    experiment_name = "yolov8s_powersocket_finetuned"

    print(f"Starting training for {epochs} epochs...")
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        optimizer=optimizer_choice,
        lr0=initial_lr,
        lrf=final_lr_factor,
        patience=early_stopping_patience,
        device=device,
        project=str(project_dir),
        name=experiment_name,
        cache="disk",
        workers=min(8, os.cpu_count() // 2 if os.cpu_count() else 1),
        cos_lr=True,
        seed=42,
        amp=True,
        augment=True,
    )

    # ----------------------------------------------------------------------
    # Training Complete - Identify and Load the Best Model
    # ----------------------------------------------------------------------
    print("\nTraining complete.")

    best_model_weights_path = Path(results.save_dir) / "weights" / "best.pt"
    best_yolo_model = model  # Fallback to the model object in memory (last epoch state)

    if best_model_weights_path.exists():
        print(
            f"Best model checkpoint (best.pt) saved by trainer at: {best_model_weights_path.resolve()}"
        )
        print("Loading this best model for validation and export...")
        best_yolo_model = YOLO(str(best_model_weights_path))
    else:
        print(
            f"ERROR: Best model checkpoint (best.pt) not found in {Path(results.save_dir) / 'weights'}."
        )
        print(
            "Proceeding with the model from the last epoch for validation/export, but this is not ideal."
        )

    # ----------------------------------------------------------------------
    # Evaluate the Best Model on the Validation Set
    # ----------------------------------------------------------------------
    print("\nEvaluating the loaded model on the validation set...")
    metrics = best_yolo_model.val(
        data=str(dataset_yaml_path),
        imgsz=image_size,
        batch=batch_size,
        device=device,
    )
    print("Validation metrics for the loaded model:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")

    # ----------------------------------------------------------------------
    # Explicitly Export the Best Model in .pt (PyTorch) Format
    # ----------------------------------------------------------------------
    final_model_filename = f"{experiment_name}_best_exported.pt"
    explicit_export_pt_path = project_dir / final_model_filename

    print(
        f"\nAttempting to explicitly export the best model to: {explicit_export_pt_path.resolve()}"
    )
    try:
        best_yolo_model.export(
            format="pt", path=str(explicit_export_pt_path), imgsz=image_size
        )
        print(
            f"Best model successfully exported to PyTorch format (.pt) at: {explicit_export_pt_path.resolve()}"
        )
        print(
            f"This file is derived from '{best_model_weights_path.resolve() if best_model_weights_path.exists() else 'last epoch model'}' and is ready for inference."
        )
    except Exception as e:
        print(f"Error during explicit .pt export of the best model: {e}")
        if best_model_weights_path.exists():
            print(
                f"You can still find the auto-saved best model (checkpoint) at: {best_model_weights_path.resolve()}"
            )

    # ----------------------------------------------------------------------
    # Optional: Perform inference on new images with the best model
    # ----------------------------------------------------------------------
    # print("\nPerforming example prediction (optional)...")
    # test_image_path = "path/to/your/test/image.jpg"
    # if Path(test_image_path).exists():
    #     predictions = best_yolo_model.predict(source=test_image_path, conf=0.25, device=device)
    #     for r in predictions:
    #         print(f"Found {len(r.boxes)} objects in {Path(r.path).name}")
    #         r.show()
    # else:
    #     print(f"Test image not found at {test_image_path}, skipping prediction example.")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
