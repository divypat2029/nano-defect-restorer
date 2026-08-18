import os
import sys
import glob
import numpy as np
import torch
from models.model_arch import SEMImageRestorer

def main():
    if len(sys.argv) < 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # 1. Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. Setup Device (Offline GPU preferred, CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Load Model and Local Weights
    model = SEMImageRestorer(in_channels=1, out_channels=1).to(device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "models", "weights.pth")
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()

    # 4. Find all .npy files in input directory
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Processing {len(npy_files)} files on {device}...")

    with torch.no_grad():
        for file_path in npy_files:
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)

            # Load input .npy array
            img_arr = np.load(file_path).astype(np.float32)

            # Clean NaNs or Infs in input
            img_arr = np.nan_to_num(img_arr, nan=0.0, posinf=1.0, neginf=0.0)

            # Normalize to [0, 1] if required
            if img_arr.max() > 1.0:
                img_arr = img_arr / 255.0
            img_arr = np.clip(img_arr, 0.0, 1.0)

            orig_h, orig_w = img_arr.shape[:2]

            # Format to Tensor: (1, 1, H, W)
            if img_arr.ndim == 2:
                tensor_in = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)
            elif img_arr.ndim == 3:
                # If (H, W, C), convert to (1, C, H, W)
                tensor_in = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)
                if tensor_in.shape[1] > 1:
                    tensor_in = tensor_in[:, :1, :, :] # Use 1st channel for grayscale
            
            tensor_in = tensor_in.to(device)

            # Inference
            restored_tensor = model(tensor_in)

            # Convert back to numpy (H, W)
            restored_arr = restored_tensor.squeeze().cpu().numpy()

            # Ensure exact target resolution
            if restored_arr.shape != (orig_h, orig_w):
                import cv2
                restored_arr = cv2.resize(restored_arr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            # Strictly enforce [0, 1] and no NaN/Inf
            restored_arr = np.nan_to_num(restored_arr, nan=0.0, posinf=1.0, neginf=0.0)
            restored_arr = np.clip(restored_arr, 0.0, 1.0).astype(np.float32)

            # Save restored array to output-dir with identical filename
            np.save(output_path, restored_arr)

    print("Restoration complete. All files saved successfully.")

if __name__ == "__main__":
    main()