"""
Konfigurasi dan utility functions untuk aplikasi Object Detection
"""

from pathlib import Path
import json
from datetime import datetime

# =====================
# PATH CONFIGURATIONS
# =====================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "runs" / "detect"
CONFIG_FILE = BASE_DIR / "app_config.json"

# =====================
# DEFAULT SETTINGS
# =====================
DEFAULT_SETTINGS = {
    "default_model": "5m.pt",
    "default_confidence": 0.25,
    "default_enhancement": None,
    "save_settings": True,
    "auto_open_result": False,
    "theme": "light",
}

# =====================
# ENHANCEMENT CONFIGS
# =====================
ENHANCEMENT_CONFIGS = {
    "hist_eq": {
        "name": "Histogram Equalization",
        "description": "Meningkatkan kontras gambar secara global",
        "best_for": "Gambar dengan pencahayaan tidak merata",
    },
    "clahe": {
        "name": "CLAHE",
        "description": "Adaptive histogram equalization",
        "best_for": "Gambar dengan detail yang kurang jelas",
        "clip_limit": 2.0,
        "tile_grid_size": (8, 8),
    },
    "brightness": {
        "name": "Brightness & Contrast",
        "description": "Menyesuaikan kecerahan dan kontras",
        "best_for": "Gambar yang terlalu gelap atau terang",
        "alpha": 1.2,  # Contrast
        "beta": 20,  # Brightness
    },
    "sharpen": {
        "name": "Sharpening",
        "description": "Mempertajam tepi dan detail",
        "best_for": "Gambar yang blur atau kurang tajam",
    },
    "denoise": {
        "name": "Denoising",
        "description": "Mengurangi noise pada gambar",
        "best_for": "Gambar dengan banyak noise atau grain",
        "h": 10,
        "template_window_size": 7,
        "search_window_size": 21,
    },
}

# =====================
# COUNTER DISPLAY CONFIGS
# =====================
COUNTER_CONFIG = {
    "font": "FONT_HERSHEY_SIMPLEX",
    "scale": 1.2,  # ← Naikkan dari 0.7 ke 1.2 (71% lebih besar)
    "thickness": 3,  # ← Naikkan dari 2 ke 3 (50% lebih tebal)
    "text_color": (0, 255, 0),  # Green (BGR)
    "bg_color": (0, 0, 0),  # Black background
    "padding": 15,  # ← Naikkan dari 10 ke 15 (padding lebih besar)
    # Position settings - RESPONSIF
    "position": "top-left",
    # Relative offset (0.0 - 1.0) berdasarkan dimensi gambar
    "offset_x_ratio": 0.015,  # ← Naikkan sedikit dari 0.01 ke 0.015
    "offset_y_ratio": 0.03,  # ← Naikkan dari 0.02 ke 0.03
    "offset_y_bottom_ratio": 0.04,  # ← Naikkan dari 0.03 ke 0.04
    # Auto-scale settings
    "use_relative": True,
    "auto_scale_text": True,
    "base_resolution": 1920,
    "min_scale": 0.6,  # ← Naikkan dari 0.4 ke 0.6 (minimum lebih besar)
    "max_scale": 1.5,  # ← Naikkan dari 1.2 ke 1.5 (maksimum lebih besar)
}


def get_counter_position(
    img_width, img_height, text_width, text_height, config=COUNTER_CONFIG
):
    """
    Hitung posisi counter berdasarkan konfigurasi RESPONSIF

    Args:
        img_width: Lebar gambar/frame
        img_height: Tinggi gambar/frame
        text_width: Lebar text yang akan ditampilkan
        text_height: Tinggi text yang akan ditampilkan
        config: Dictionary konfigurasi counter

    Returns:
        (x, y, scale_factor): Koordinat untuk cv2.putText + scale factor untuk text
    """
    position = config.get("position", "top-left")
    use_relative = config.get("use_relative", True)

    # Calculate scale factor berdasarkan resolusi
    scale_factor = 1.0
    if config.get("auto_scale_text", True):
        base_res = config.get("base_resolution", 1920)
        min_scale = config.get("min_scale", 0.4)
        max_scale = config.get("max_scale", 1.2)

        # Scale berdasarkan width (works untuk portrait & landscape)
        scale_factor = img_width / base_res
        scale_factor = max(min_scale, min(max_scale, scale_factor))

    # Calculate offsets
    if use_relative:
        # Gunakan ratio (responsif!)
        offset_x = int(img_width * config.get("offset_x_ratio", 0.01))

        if "bottom" in position:
            offset_y = int(img_height * config.get("offset_y_bottom_ratio", 0.03))
        else:
            offset_y = int(img_height * config.get("offset_y_ratio", 0.02))
    else:
        # Gunakan fixed pixel
        offset_x = config.get("offset_x_px", 10)
        offset_y = config.get("offset_y_px", 30)

    # Calculate position berdasarkan corner
    if position == "top-left":
        x = offset_x
        y = offset_y + text_height
    elif position == "top-right":
        x = img_width - text_width - offset_x - config.get("padding", 10)
        y = offset_y + text_height
    elif position == "bottom-left":
        x = offset_x
        y = img_height - offset_y
    elif position == "bottom-right":
        x = img_width - text_width - offset_x - config.get("padding", 10)
        y = img_height - offset_y
    else:
        # Default top-left
        x = offset_x
        y = offset_y + text_height

    return (x, y, scale_factor)


# =====================
# SUPPORTED FORMATS
# =====================
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]

# =====================
# MODEL CONFIGURATIONS
# =====================
MODEL_INFO = {
    "5n.pt": {
        "name": "YOLOv5 Nano",
        "size": "1.9 MB",
        "speed": "Very Fast",
        "accuracy": "Low-Medium",
        "description": "Model paling ringan, cocok untuk real-time",
    },
    "5s.pt": {
        "name": "YOLOv5 Small",
        "size": "7.2 MB",
        "speed": "Fast",
        "accuracy": "Medium",
        "description": "Balance antara kecepatan dan akurasi",
    },
    "5m.pt": {
        "name": "YOLOv5 Medium",
        "size": "21.2 MB",
        "speed": "Medium",
        "accuracy": "Medium-High",
        "description": "Pilihan terbaik untuk kebanyakan kasus",
    },
    "5l.pt": {
        "name": "YOLOv5 Large",
        "size": "46.5 MB",
        "speed": "Slow",
        "accuracy": "High",
        "description": "Akurasi tinggi, butuh resource lebih besar",
    },
    "11m.pt": {
        "name": "YOLOv11 Medium",
        "size": "~20 MB",
        "speed": "Medium",
        "accuracy": "High",
        "description": "Model YOLOv11 versi medium",
    },
    "11n.pt": {
        "name": "YOLOv11 Nano",
        "size": "~3 MB",
        "speed": "Very Fast",
        "accuracy": "Medium",
        "description": "Model YOLOv11 versi nano",
    },
    "11s.pt": {
        "name": "YOLOv11 Small",
        "size": "~10 MB",
        "speed": "Fast",
        "accuracy": "Medium-High",
        "description": "Model YOLOv11 versi small",
    },
}

# =====================
# UTILITY FUNCTIONS
# =====================


def load_config():
    """Load konfigurasi dari file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                # Merge dengan default settings
                return {**DEFAULT_SETTINGS, **config}
        except Exception as e:
            print(f"Warning: Tidak bisa load config: {e}")
    return DEFAULT_SETTINGS.copy()


def save_config(config):
    """Simpan konfigurasi ke file"""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Warning: Tidak bisa save config: {e}")
        return False


def get_available_models():
    """Dapatkan list model yang tersedia"""
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return []

    models = []
    for model_file in MODEL_DIR.glob("*.pt"):
        model_name = model_file.name
        info = MODEL_INFO.get(
            model_name,
            {
                "name": model_name,
                "size": "Unknown",
                "speed": "Unknown",
                "accuracy": "Unknown",
                "description": "Custom model",
            },
        )
        models.append({"filename": model_name, "path": str(model_file), "info": info})

    return models


def create_output_filename(source_path, prefix="result"):
    """Generate output filename dengan timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = Path(source_path).stem
    ext = Path(source_path).suffix
    return f"{prefix}_{source_name}_{timestamp}{ext}"


def is_image(file_path):
    """Cek apakah file adalah gambar"""
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_FORMATS


def is_video(file_path):
    """Cek apakah file adalah video"""
    return Path(file_path).suffix.lower() in SUPPORTED_VIDEO_FORMATS


def get_file_size(file_path):
    """Dapatkan ukuran file dalam format readable"""
    size_bytes = Path(file_path).stat().st_size

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.2f} TB"


def validate_confidence(conf_value):
    """Validasi nilai confidence threshold"""
    try:
        conf = float(conf_value)
        return max(0.01, min(1.0, conf))
    except:
        return 0.25


def get_enhancement_info(enhancement_type):
    """Dapatkan informasi tentang enhancement"""
    return ENHANCEMENT_CONFIGS.get(enhancement_type, None)


def format_duration(seconds):
    """Format durasi dalam format readable"""
    if seconds < 60:
        return f"{seconds:.2f} detik"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} menit {int(secs)} detik"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)} jam {int(minutes)} menit"


def ensure_directories():
    """Pastikan semua direktori yang diperlukan ada"""
    directories = [MODEL_DIR, OUTPUT_DIR]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# =====================
# LOG UTILITIES
# =====================


class Logger:
    """Simple logger untuk tracking process"""

    def __init__(self, log_file=None):
        self.log_file = log_file
        self.logs = []

    def log(self, message, level="INFO"):
        """Add log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)

        print(log_entry)

        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(log_entry + "\n")
            except:
                pass

    def info(self, message):
        """Log info message"""
        self.log(message, "INFO")

    def warning(self, message):
        """Log warning message"""
        self.log(message, "WARNING")

    def error(self, message):
        """Log error message"""
        self.log(message, "ERROR")

    def success(self, message):
        """Log success message"""
        self.log(message, "SUCCESS")

    def get_logs(self):
        """Get all logs"""
        return "\n".join(self.logs)

    def clear(self):
        """Clear logs"""
        self.logs = []


# =====================
# INITIALIZATION
# =====================

# Pastikan direktori ada saat module di-import
ensure_directories()

# Export semua yang penting
__all__ = [
    "MODEL_DIR",
    "OUTPUT_DIR",
    "DEFAULT_SETTINGS",
    "ENHANCEMENT_CONFIGS",
    "SUPPORTED_IMAGE_FORMATS",
    "SUPPORTED_VIDEO_FORMATS",
    "MODEL_INFO",
    "load_config",
    "save_config",
    "get_available_models",
    "create_output_filename",
    "is_image",
    "is_video",
    "get_file_size",
    "validate_confidence",
    "get_enhancement_info",
    "format_duration",
    "Logger",
]
