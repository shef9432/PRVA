import unittest

import cv2
import numpy as np

from pv_audit import AuditAgent, BaseAdapter
from pv_audit.stress import StressSimulator


def textured(h=120, w=160):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.putText(img, "PVA", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return img


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


class TestDpi(unittest.TestCase):
    def test_level_zero_is_identity(self):
        img = textured()
        self.assertTrue(np.array_equal(StressSimulator.apply_dpi(img, 0), img))

    def test_keeps_size_and_loses_detail_monotonically(self):
        img = textured()
        out = [StressSimulator.apply_dpi(img, v) for v in (20, 50, 80)]
        for o in out:
            self.assertEqual(o.shape, img.shape)
            self.assertEqual(o.dtype, np.uint8)
        s = [sharpness(img)] + [sharpness(o) for o in out]
        self.assertTrue(all(a > b for a, b in zip(s, s[1:])), s)

    def test_extreme_level_still_valid(self):
        self.assertEqual(StressSimulator.apply_dpi(textured(), 100).shape, (120, 160, 3))


class TestDefocus(unittest.TestCase):
    def test_level_zero_is_identity(self):
        img = textured()
        self.assertTrue(np.array_equal(StressSimulator.apply_defocus(img, 0), img))

    def test_blur_grows_with_level(self):
        img = textured()
        s = [sharpness(StressSimulator.apply_defocus(img, v)) for v in (0, 10, 30, 60)]
        self.assertTrue(all(a > b for a, b in zip(s, s[1:])), s)

    def test_keeps_mean_brightness(self):
        img = textured()
        self.assertAlmostEqual(StressSimulator.apply_defocus(img, 40).mean(), img.mean(), delta=2.0)


class Echo(BaseAdapter):
    def predict(self, image):
        return {"data": None, "conf": float(sharpness(image))}


class TestEngine(unittest.TestCase):
    def test_new_dimensions_in_an_audit(self):
        report = AuditAgent(Echo()).run(textured(), dims=["DPI", "Defocus"], lvls=[0, 40])
        df = report.df
        self.assertEqual(sorted(df["dimension"].unique()), ["DPI", "Defocus"])
        for dim in ("DPI", "Defocus"):
            c = df[df["dimension"] == dim].set_index("level")["conf"]
            self.assertGreater(c[0], c[40])


if __name__ == "__main__":
    unittest.main()
