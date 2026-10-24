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


def stripes(vertical: bool, h=120, w=160, period=8):
    img = np.zeros((h, w, 3), np.uint8)
    for k in range(0, w if vertical else h, period):
        if vertical:
            img[:, k:k + period // 2] = 255
        else:
            img[k:k + period // 2, :] = 255
    return img


def contrast(img):
    return float(img.astype(np.float32).std())


class TestMotionBlur(unittest.TestCase):
    def test_level_zero_is_identity(self):
        img = textured()
        self.assertTrue(np.array_equal(StressSimulator.apply_mblur(img, 0), img))

    def test_directional_horizontal_by_default(self):
        v, h = stripes(vertical=True), stripes(vertical=False)
        bv, bh = StressSimulator.apply_mblur(v, 40), StressSimulator.apply_mblur(h, 40)
        self.assertLess(contrast(bv), 0.5 * contrast(v))         # smeared across vertical stripes
        self.assertGreater(contrast(bh), 0.95 * contrast(h))     # stripes parallel to the motion survive

    def test_angle(self):
        h = stripes(vertical=False)
        self.assertLess(contrast(StressSimulator.apply_mblur(h, 40, angle=90)), 0.5 * contrast(h))

    def test_stronger_with_level(self):
        v = stripes(vertical=True, period=16)
        c = [contrast(StressSimulator.apply_mblur(v, x)) for x in (0, 20, 40, 80)]
        self.assertTrue(all(a > b for a, b in zip(c, c[1:])), c)


class TestLux(unittest.TestCase):
    def test_level_zero_is_identity(self):
        img = textured()
        for f in (StressSimulator.apply_lux_plus, StressSimulator.apply_lux_minus):
            self.assertTrue(np.array_equal(f(img, 0), img))

    def test_brighter_and_darker_monotonic(self):
        img = np.full((10, 10, 3), 100, np.uint8)
        up = [StressSimulator.apply_lux_plus(img, v).mean() for v in (0, 20, 40)]
        dn = [StressSimulator.apply_lux_minus(img, v).mean() for v in (0, 20, 40)]
        self.assertTrue(up[0] < up[1] < up[2], up)
        self.assertTrue(dn[0] > dn[1] > dn[2], dn)

    def test_gamma_not_just_linear(self):
        img = np.full((4, 4, 3), 64, np.uint8)
        gain = 1 + 20 / 50
        self.assertGreater(StressSimulator.apply_lux_plus(img, 20).mean(), 64 * gain + 1)   # midtones lifted
        self.assertLess(StressSimulator.apply_lux_minus(img, 20).mean(), 64 / gain - 1)     # shadows crushed

    def test_lux_minus_in_engine(self):
        report = AuditAgent(Echo()).run(textured(), dims=["Lux-", "MBlur"], lvls=[0, 40])
        self.assertEqual(sorted(report.df["dimension"].unique()), ["Lux-", "MBlur"])


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
