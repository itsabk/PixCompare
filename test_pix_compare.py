import unittest
import cv2
import numpy as np
from pix_compare import compare_images, align_images, highlight_differences
from colorama import init, Fore, Style
import time

init(autoreset=True)  # Initialize colorama for Windows

total_tests = 0
passed_tests = 0
failed_tests = 0
skipped_tests = 0

class TestImageComparison(unittest.TestCase):
    def setUp(self):
        self.image1 = cv2.imread('image1.jpg')
        self.image2 = cv2.imread('image2.jpg')
        self.methods = ['SIFT', 'BRISK', 'AKAZE', 'KAZE', 'BRIEF', 'FREAK', 'LATCH', 'LUCID', 'DAISY', 'ORB']

    def test_align_images(self):
        global total_tests, passed_tests, failed_tests, skipped_tests
        for method in self.methods:
            total_tests += 1
            with self.subTest(method=method):
                try:
                    detector = cv2.SIFT_create() if method == 'SIFT' else cv2.ORB_create()
                    aligned_image, _, success, _ = align_images(self.image1, self.image2, detector)
                    self.assertTrue(success)
                    self.assertIsInstance(aligned_image, np.ndarray)

                    aligned_image, _, success, crop_coords = align_images(self.image1, self.image2, detector, auto_crop=True)
                    self.assertTrue(success)
                    self.assertIsInstance(aligned_image, np.ndarray)
                    self.assertIsInstance(crop_coords, tuple)
                    passed_tests += 1
                except cv2.error as e:
                    if "The function/feature is not implemented" in str(e):
                        print(f"{Fore.YELLOW}Skipping {method} due to unsupported feature detection method.{Style.RESET_ALL}")
                        skipped_tests += 1
                    else:
                        failed_tests += 1
                        raise e

    def test_highlight_differences(self):
        global total_tests, passed_tests
        total_tests += 2
        highlighted_image = highlight_differences(self.image1, self.image2)
        self.assertIsInstance(highlighted_image, np.ndarray)
        passed_tests += 1

        highlighted_image = highlight_differences(self.image1, self.image2, sensitivity_threshold=30, blur_value=(15, 15))
        self.assertIsInstance(highlighted_image, np.ndarray)
        passed_tests += 1

    def test_compare_images(self):
        global total_tests, passed_tests, failed_tests, skipped_tests
        for method in self.methods:
            total_tests += 1
            with self.subTest(method=method):
                print(f"{Fore.GREEN}Testing with method: {method}{Style.RESET_ALL}")
                try:
                    start_time = time.time()
                    compare_images('image1.jpg', 'image2.jpg', method=method, align=False)
                    compare_images('image1.jpg', 'image2.jpg', method=method, align=True, auto_crop=True)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"{Fore.CYAN}Execution time for {method}: {execution_time:.2f} seconds{Style.RESET_ALL}")
                    passed_tests += 1
                except cv2.error as e:
                    if "The function/feature is not implemented" in str(e):
                        print(f"{Fore.YELLOW}Skipping {method} due to unsupported feature detection method.{Style.RESET_ALL}")
                        skipped_tests += 1
                    else:
                        failed_tests += 1
                        raise e

        total_tests += 2
        with self.assertRaises(ValueError):
            compare_images('image1.jpg', 'image2.jpg', method='INVALID_METHOD')
            passed_tests += 1

        with self.assertRaises(ValueError):
            compare_images('invalid/path.jpg', 'image2.jpg')
            passed_tests += 1


if __name__ == '__main__':
    print(f"{Fore.CYAN}Running tests...{Style.RESET_ALL}")
    start_time = time.time()
    unittest.main(exit=False)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{Fore.CYAN}Test summary:{Style.RESET_ALL}")
    print(f"Total tests: {total_tests}")
    print(f"{Fore.GREEN}Passed tests: {passed_tests}{Style.RESET_ALL}")
    print(f"{Fore.RED}Failed tests: {failed_tests}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Skipped tests: {skipped_tests}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}Total time taken: {elapsed_time:.2f} seconds{Style.RESET_ALL}")