import cv2
import numpy as np

class StressSimulator:
    """Physics-based degradation operators for industrial environments."""
    
    @staticmethod
    def apply_lux_plus(img: np.ndarray, val: int) -> np.ndarray:
        return np.clip(img.astype(np.float32) * (1 + val / 50), 0, 255).astype(np.uint8)

    @staticmethod
    def apply_mblur(img: np.ndarray, val: int) -> np.ndarray:
        kernel_size = int(val / 10) * 2 + 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) if kernel_size > 1 else img

    @staticmethod
    def apply_iso_noise(img: np.ndarray, val: int) -> np.ndarray:
        noise = np.random.normal(0, val, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_dpi(img: np.ndarray, val: int) -> np.ndarray:
        """Working distance / resolution loss: area-based downsampling to (100 - val)% of the pixels per side,
        then back to the original size. 0 = unchanged; at least 2 pixels per side remain."""
        scale = max(0.0, 1.0 - val / 100)
        if scale >= 1.0:
            return img
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def apply_defocus(img: np.ndarray, val: int) -> np.ndarray:
        """Focal drift: Gaussian blur with sigma = val / 10 pixels. 0 = unchanged."""
        sigma = val / 10
        if sigma <= 0:
            return img
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
