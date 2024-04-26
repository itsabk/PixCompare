import unittest
import cv2
import numpy as np
from pix_compare import compare_images, align_images, highlight_differences

class TestImageComparison(unittest.TestCase):
    def setUp(self):
        self.image1 = cv2.imread('image1.jpg')
        self.image2 = cv2.imread('image2.jpg')
        self.methods = ['SIFT', 'BRISK', 'AKAZE', 'KAZE', 'BRIEF', 'FREAK', 'LATCH', 'LUCID', 'DAISY', 'ORB']

    def test_highlight_differences(self):
        # Test with default parameters
        highlighted_image = highlight_differences(self.image1, self.image2)
        self.assertTrue(isinstance(highlighted_image, str))

        # Test with custom sensitivity_threshold and blur_value
        highlighted_image = highlight_differences(self.image1, self.image2, sensitivity_threshold=30, blur_value=(15, 15))
        self.assertTrue(isinstance(highlighted_image, str))

    def test_align_images(self):
        for method in self.methods:
            detector = cv2.__dict__[f'{method}_create']()
            with self.subTest(method=method):
                # Test without auto_crop
                aligned_image, homography, success = align_images(self.image1, self.image2, detector)
                self.assertTrue(success)
                self.assertIsNotNone(homography)

                # Test with auto_crop
                aligned_image, homography, success = align_images(self.image1, self.image2, detector, auto_crop=True)
                self.assertTrue(success)
                self.assertIsNotNone(homography)

    def test_compare_images(self):
        for method in self.methods:
            with self.subTest(method=method):
                # Test without alignment
                compare_images('image1.jpg', 'image2.jpg', method=method, align=False)

                # Test with alignment and auto_crop
                compare_images('image1.jpg', 'image2.jpg', method=method, align=True, auto_crop=True)

        # Test with invalid method
        with self.assertRaises(ValueError):
            compare_images('image1.jpg', 'image2.jpg', method='INVALID_METHOD')

        # Test with invalid image paths
        with self.assertRaises(ValueError):
            compare_images('invalid/path.jpg', 'image2.jpg')

if __name__ == '__main__':
    unittest.main()