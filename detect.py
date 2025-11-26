import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse
import traceback
import time
import shutil

# Import config
try:
    from config import COUNTER_CONFIG, get_counter_position
except ImportError:
    # Fallback jika config.py tidak ada - RESPONSIVE VERSION dengan UKURAN LEBIH BESAR
    COUNTER_CONFIG = {
        "scale": 1.2,  # ← Lebih besar dari 0.7
        "thickness": 3,  # ← Lebih tebal dari 2
        "text_color": (0, 255, 0),
        "bg_color": (0, 0, 0),
        "padding": 15,  # ← Padding lebih besar
        "position": "top-left",
        "offset_x_ratio": 0.015,
        "offset_y_ratio": 0.03,
        "offset_y_bottom_ratio": 0.04,
        "use_relative": True,
        "auto_scale_text": True,
        "base_resolution": 1920,
        "min_scale": 0.6,  # ← Minimum scale lebih besar
        "max_scale": 1.5,  # ← Maximum scale lebih besar
    }

    def get_counter_position(
        img_width, img_height, text_width, text_height, config=COUNTER_CONFIG
    ):
        """Fallback responsive position calculator"""
        position = config.get("position", "top-left")

        # Scale factor
        scale_factor = 1.0
        if config.get("auto_scale_text", True):
            scale_factor = img_width / config.get("base_resolution", 1920)
            scale_factor = max(
                config.get("min_scale", 0.6),
                min(config.get("max_scale", 1.5), scale_factor),
            )

        # Offsets
        offset_x = int(img_width * config.get("offset_x_ratio", 0.015))

        if "bottom" in position:
            offset_y = int(img_height * config.get("offset_y_bottom_ratio", 0.04))
            y = img_height - offset_y
        else:
            offset_y = int(img_height * config.get("offset_y_ratio", 0.03))
            y = offset_y + text_height

        x = offset_x

        return (x, y, scale_factor)


CLASS_LABELS = {
    0: "ayam",
    1: "babi",
    2: "kambing",
    3: "bebek",
    4: "sapi",
}


def apply_enhancement(image, enhancement_type):
    """
    Aplikasikan berbagai jenis enhancement pada gambar

    Args:
        image: Input image (BGR format)
        enhancement_type: Jenis enhancement ('hist_eq', 'clahe', 'brightness', 'sharpen', 'denoise', None)

    Returns:
        Enhanced image
    """
    if enhancement_type is None:
        return image

    if enhancement_type == "hist_eq":
        # Histogram Equalization
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    elif enhancement_type == "clahe":
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    elif enhancement_type == "brightness":
        # Brightness & Contrast adjustment
        alpha = 1.2  # Contrast control (1.0-3.0)
        beta = 20  # Brightness control (0-100)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    elif enhancement_type == "sharpen":
        # Sharpening using kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    elif enhancement_type == "denoise":
        # Denoising
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return image


def load_model(weights_path, device="cpu"):
    """
    Load YOLOv5/YOLOv11 model dengan dukungan GPU/CPU
    """
    print(f"Loading model dari: {weights_path}")
    print(f"Device request dari GUI: {device}")

    # Normalize device strings: 'cuda' -> 'cuda:0', digit -> 'cuda:N'
    if isinstance(device, str):
        if device.lower() == "cuda":
            device = "cuda:0"
        elif device.isdigit():
            device = f"cuda:{device}"

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print("[debug] CUDA tersedia:", cuda_available)
    print(f"[debug] normalized device (requested): {device}")

    # If requested GPU but CUDA not available -> fallback to CPU
    if isinstance(device, str) and device.startswith("cuda") and not cuda_available:
        print("⚠ CUDA tidak tersedia, fallback ke CPU")
        device = "cpu"

    # Deteksi YOLOv11 atau YOLOv5
    model_name = Path(weights_path).name.lower()
    is_v11 = "11" in model_name or "v11" in model_name

    try:
        if is_v11:
            # YOLOv11 - Import dari submodule
            from ultralytics.models import YOLO

            print("Loading YOLOv11 model...")
            t0 = time.time()
            model = YOLO(weights_path)
            print(f"[debug] YOLOv11 model instantiation time: {time.time()-t0:.2f}s")

            # Move model to device
            try:
                if isinstance(device, str) and device.startswith("cuda"):
                    print(f"[debug] Attempting to move YOLOv11 model to {device}")
                    model.to(device)
                    print(f"✓ YOLOv11 loaded on GPU ({device})")
                else:
                    print("[debug] Moving YOLOv11 model to cpu")
                    model.to("cpu")
                    print("✓ YOLOv11 loaded on CPU")
            except Exception as e:
                print(f"[error] Failed to move YOLOv11 to GPU: {e}")
                traceback.print_exc()
                model.to("cpu")
                print("⚠ Gagal memindahkan YOLOv11 ke GPU, menggunakan CPU")

            return model

        else:
            # YOLOv5
            print("Loading YOLOv5 model via torch.hub...")
            t0 = time.time()
            # Convert Path to string for torch.hub.load (torch.hub doesn't handle Path objects well)
            weights_str = str(weights_path)
            model = None

            # Attempt 1: Normal load
            try:
                model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=weights_str,
                    force_reload=False,
                    verbose=False,
                )
            except Exception as e:
                if "pathlib._local" in str(e) or "pathlib" in str(e):
                    print(f"[warn] Cache corruption detected: {e}")
                    print("[debug] Clearing torch hub cache and retrying...")
                    # Clear torch hub cache
                    import shutil

                    hub_dir = Path.home() / ".cache" / "torch" / "hub"
                    if hub_dir.exists():
                        try:
                            shutil.rmtree(hub_dir)
                            print("[debug] Cache cleared successfully")
                        except Exception as ce:
                            print(f"[warn] Could not clear cache: {ce}")
                else:
                    print(f"[warn] First load attempt failed: {e}")

                # Attempt 2: Force reload
                print("[debug] Retrying with force_reload=True...")
                try:
                    model = torch.hub.load(
                        "ultralytics/yolov5",
                        "custom",
                        path=weights_str,
                        force_reload=True,
                        verbose=False,
                    )
                except Exception as e2:
                    print(f"[error] Second attempt failed: {e2}")
                    raise

            print(f"[debug] YOLOv5 torch.hub.load time: {time.time()-t0:.2f}s")

            # Move model to deviceI
            try:
                if isinstance(device, str) and device.startswith("cuda"):
                    print(f"[debug] Attempting to move YOLOv5 model to {device}")
                    model.to(device)  # type: ignore
                    print(f"✓ YOLOv5 loaded on GPU ({device})")
                else:
                    print("[debug] Moving YOLOv5 model to cpu")
                    model.to("cpu")  # type: ignore
                    print("✓ YOLOv5 loaded on CPU")
            except Exception as e:
                print(f"[error] Failed to move YOLOv5 to GPU: {e}")
                traceback.print_exc()
                model.to("cpu")  # type: ignore
                print("⚠ Gagal memindahkan YOLOv5 ke GPU, menggunakan CPU")

            return model

    except Exception as e:
        print("Error loading model:", e)
        raise


def detect_image(model, image_path, conf_thres=0.25, enhancement=None):
    """
    Deteksi objek pada gambar (support YOLOv5 dan YOLOv11)

    Args:
        model: YOLOv5/YOLOv11 model
        image_path: Path ke gambar input
        conf_thres: Confidence threshold
        enhancement: Jenis enhancement

    Returns:
        Annotated image (numpy array)
    """
    # Baca gambar
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Tidak dapat membaca gambar: {image_path}")

    # Apply enhancement jika ada
    if enhancement:
        img = apply_enhancement(img, enhancement)

    # Cek tipe model (YOLOv5 atau YOLOv11)
    model_type = type(model).__name__

    if "YOLO" in model_type and hasattr(model, "predict"):
        # YOLOv11 (ultralytics)
        results = model.predict(img, conf=conf_thres, verbose=False)
        annotated_img = results[0].plot()
    else:
        # YOLOv5
        model.conf = conf_thres
        results = model(img)
        annotated_img = results.render()[0]

    return annotated_img


def detect_video(model, video_path, output_path, conf_thres=0.25, enhancement=None):
    """
    Deteksi objek pada video (support YOLOv5 dan YOLOv11)

    Args:
        model: YOLOv5/YOLOv11 model
        video_path: Path ke video input
        output_path: Path untuk menyimpan video output
        conf_thres: Confidence threshold
        enhancement: Jenis enhancement

    Returns:
        Path ke video output
    """
    # Buka video input
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Tidak dapat membuka video: {video_path}")

    # Dapatkan properties video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Cek tipe model
    model_type = type(model).__name__
    is_v11 = "YOLO" in model_type and hasattr(model, "predict")

    if not is_v11:
        # Set confidence threshold untuk YOLOv5
        model.conf = conf_thres

    frame_count = 0
    print(f"Processing video: {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply enhancement jika ada
        if enhancement:
            frame = apply_enhancement(frame, enhancement)

        # Inference
        if is_v11:
            # YOLOv11
            results = model.predict(frame, conf=conf_thres, verbose=False)
            annotated_frame = results[0].plot()
        else:
            # YOLOv5
            results = model(frame)
            annotated_frame = results.render()[0]

        # Tulis frame ke output
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(
                f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)"
            )

    # Release resources
    cap.release()
    out.release()

    print(f"Video processing complete! Saved to: {output_path}")
    return output_path


def get_count_text(results, model):
    """
    Hitung jumlah objek per kelas dan kembalikan teks ringkas.
    Support:
      - ultralytics YOLO (results.boxes.cls)
      - YOLOv5 hub (results.xyxy[0][:, -1])
    """
    cls_ids = []

    # ultralytics YOLO (v8/v11): ada .boxes
    if hasattr(results, "boxes") and results.boxes is not None:
        if len(results.boxes) == 0:
            return "Tidak ada objek terdeteksi"
        cls_ids = results.boxes.cls.int().tolist()

        names = getattr(results, "names", None) or getattr(model, "names", None) or {}

    else:
        # YOLOv5 hub: gunakan xyxy[0], kolom terakhir = class id
        if (
            not hasattr(results, "xyxy")
            or len(results.xyxy) == 0
            or results.xyxy[0] is None
        ):
            return "Tidak ada objek terdeteksi"
        cls_ids = results.xyxy[0][:, -1].int().tolist()
        names = getattr(model, "names", {})

    counter = Counter(cls_ids)
    total = sum(counter.values())
    parts = []

    for cls_id, num in counter.items():
        name = (
            names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        )
        parts.append(f"{name}: {num}")

    return f"Total: {total} | " + ", ".join(parts)


def draw_counter_text(image, count_text, config=COUNTER_CONFIG):
    """
    Gambar counter text pada image dengan konfigurasi RESPONSIF

    Args:
        image: Image/frame (numpy array)
        count_text: Text yang akan ditampilkan
        config: Dictionary konfigurasi counter

    Returns:
        Image dengan counter text
    """
    # Ensure image is writeable
    image = np.ascontiguousarray(image).copy()

    img_height, img_width = image.shape[:2]

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = config.get("scale", 0.7)
    base_thickness = config.get("thickness", 2)
    text_color = config.get("text_color", (0, 255, 0))
    bg_color = config.get("bg_color", (0, 0, 0))
    padding = config.get("padding", 10)

    # Get text size dengan base scale
    (text_w, text_h), baseline = cv2.getTextSize(
        count_text, font, base_scale, base_thickness
    )

    # Get responsive position dan scale factor
    x, y, scale_factor = get_counter_position(
        img_width, img_height, text_w, text_h, config
    )

    # Apply scale factor ke text size dan thickness
    final_scale = base_scale * scale_factor
    final_thickness = max(1, int(base_thickness * scale_factor))

    # Recalculate text size dengan final scale
    (text_w, text_h), baseline = cv2.getTextSize(
        count_text, font, final_scale, final_thickness
    )

    # Adjust padding based on scale
    scaled_padding = int(padding * scale_factor)

    # Draw background rectangle
    position = config.get("position", "top-left")
    if "bottom" in position:
        # Bottom position - rectangle above text
        rect_top = y - text_h - scaled_padding
        rect_bottom = y + baseline
        rect_left = x
        rect_right = x + text_w + scaled_padding * 2
    else:
        # Top position - rectangle around text
        rect_top = y - text_h - scaled_padding // 2
        rect_bottom = y + baseline + scaled_padding // 2
        rect_left = x
        rect_right = x + text_w + scaled_padding * 2

    # Ensure rectangle stays within image bounds
    rect_top = max(0, rect_top)
    rect_bottom = min(img_height, rect_bottom)
    rect_left = max(0, rect_left)
    rect_right = min(img_width, rect_right)

    cv2.rectangle(image, (rect_left, rect_top), (rect_right, rect_bottom), bg_color, -1)

    # Draw text
    text_x = x + scaled_padding
    text_y = y if "bottom" not in position else y - scaled_padding // 2

    # Ensure text position is within bounds
    text_x = max(0, min(img_width - text_w, text_x))
    text_y = max(text_h, min(img_height, text_y))

    cv2.putText(
        image,
        count_text,
        (text_x, text_y),
        font,
        final_scale,
        text_color,
        final_thickness,
        cv2.LINE_AA,
    )

    return image


def run(
    weights,
    source,
    conf_thres=0.25,
    enhancement=None,
    output_dir="runs/detect",
    device="cpu",
    counter_config=None,  # ← Tambah parameter baru
):
    """
    Fungsi utama untuk menjalankan deteksi dengan dukungan GPU/CPU + counting.
    """
    # Use default config if not provided
    if counter_config is None:
        counter_config = COUNTER_CONFIG

    # Buat direktori output jika belum ada
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # === Normalisasi / cek device ===
    if isinstance(device, str):
        if device.lower() == "cuda":
            device = "cuda:0"
        elif device.isdigit():
            device = f"cuda:{device}"

    if isinstance(device, str) and device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("⚠ Warning: CUDA tidak tersedia, menggunakan CPU")
            device = "cpu"
        else:
            try:
                idx = int(device.split(":")[1]) if ":" in device else 0
                torch.cuda.set_device(idx)
                print(f"✓ Using GPU: {torch.cuda.get_device_name(idx)}")
            except Exception as e:
                print(
                    f"⚠ Warning: Tidak dapat mengatur GPU {device} ({e}), menggunakan CPU"
                )
                device = "cpu"

    # === Load model ===
    print(f"Loading model from: {weights}")
    model = load_model(weights, device)

    source_path = Path(source)
    file_ext = source_path.suffix.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =========================================================
    #                       G A M B A R
    # =========================================================
    if file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        print(f"Detecting objects in image: {source}")
        if enhancement:
            print(f"Applying enhancement: {enhancement}")

        img = cv2.imread(str(source_path))
        if img is None:
            raise ValueError(f"Tidak dapat membaca gambar: {source_path}")

        if enhancement:
            img = apply_enhancement(img, enhancement)

        # Inference YOLO
        try:
            # ultralytics YOLO (v8/11)
            results_list = model.predict(img, conf=conf_thres, verbose=False)
            results = results_list[0]
            annotated = results.plot()
        except AttributeError:
            # YOLOv5 hub
            model.conf = conf_thres
            results = model(img)
            annotated = results.render()[0]

        # Get counting text
        count_text = get_count_text(results, model)

        # Draw counter dengan config yang bisa disesuaikan
        annotated = draw_counter_text(annotated, count_text, counter_config)

        output_file = output_path / f"result_{timestamp}.jpg"
        cv2.imwrite(str(output_file), annotated)
        print(f"Results saved to: {output_file}")
        return str(output_file)

    # =========================================================
    #                        V I D E O
    # =========================================================
    elif file_ext in [".mp4", ".avi", ".mov", ".mkv"]:
        print(f"Detecting objects in video: {source}")
        if enhancement:
            print(f"Applying enhancement: {enhancement}")

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise ValueError(f"Tidak dapat membuka video: {source_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        output_file = output_path / f"result_{timestamp}.mp4"
        out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

        frame_count = 0
        print("Processing video...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if enhancement:
                frame = apply_enhancement(frame, enhancement)

            try:
                results_list = model.predict(frame, conf=conf_thres, verbose=False)
                results = results_list[0]
                annotated = results.plot()
            except AttributeError:
                model.conf = conf_thres
                results = model(frame)
                annotated = results.render()[0]

            # Get counting text
            count_text = get_count_text(results, model)

            # Draw counter dengan config yang bisa disesuaikan
            annotated = draw_counter_text(annotated, count_text, counter_config)

            out.write(annotated)
            frame_count += 1

        cap.release()
        out.release()
        print(f"Video processing complete! Saved to: {output_file}")
        return str(output_file)

    else:
        raise ValueError(f"Format file tidak didukung: {file_ext}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="YOLOv5/YOLOv11 Object Detection dengan Enhancement dan GPU Support"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path ke model weights (.pt file)"
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Path ke gambar atau video input"
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--enhancement",
        type=str,
        choices=["hist_eq", "clahe", "brightness", "sharpen", "denoise"],
        help="Jenis image enhancement",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/detect", help="Direktori output"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu, cuda, 0, 1, etc.)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Jika dijalankan dari command line
    args = parse_args()

    try:
        output_path = run(
            weights=args.weights,
            source=args.source,
            conf_thres=args.conf_thres,
            enhancement=args.enhancement,
            output_dir=args.output_dir,
            device=args.device,
        )
        print(f"\n✓ Detection completed successfully!")
        print(f"Output: {output_path}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
