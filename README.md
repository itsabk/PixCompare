# PixCompare

PixCompare is a Python tool designed to highlight unique features between two images, making it easier to visually identify differences. This tool is particularly useful for comparing images that may not be perfectly aligned or are of different sizes.

## Features

- **Image Alignment**: Aligns two images using various feature detection methods (SIFT, BRISK, AKAZE, KAZE, BRIEF, FREAK, LATCH, LUCID, DAISY, or ORB) and homography before comparing them.
- **Image Resizing**: Automatically resizes images to the same dimensions for accurate comparison.
- **Gaussian Blur Application**: Applies Gaussian blur to both images to focus on significant differences.
- **Color Highlighted Differences**: Displays differences in color on a grayscale background, making them easy to identify.
- **Auto Cropping**: Automatically removes black borders introduced during alignment for a cleaner output.

## Getting Started

### Dependencies

- Python 3.x
- OpenCV
- NumPy

### Installing

1. Clone the repository:
   ```bash
   git clone https://github.com/itsabk/PixCompare.git
   ```
2. Navigate to the project directory:
   ```bash
   cd PixCompare
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To utilize PixCompare, follow these steps:

1. **Prepare Your Images**: Ensure you have two images that you wish to compare.

2. **Use the `compare_images` Function**: This function is central to PixCompare. It offers options for aligning images before highlighting differences.

   ```python
   from pix_compare import compare_images

   # Basic usage without image alignment
   compare_images('path/to/image1.jpg', 'path/to/image2.jpg')

   # With image alignment using the ORB feature detection method
   compare_images('path/to/image1.jpg', 'path/to/image2.jpg', align=True)

   # Advanced usage with custom settings for feature detection method, sensitivity, and blur
   compare_images(
       'path/to/image1.jpg',
       'path/to/image2.jpg',
       method='SIFT', # Choose from SIFT, BRISK, AKAZE, KAZE, BRIEF, FREAK, LATCH, LUCID, DAISY, or ORB
       align=True,
       sensitivity_threshold=40,  # Adjust the sensitivity of difference highlighting
       blur_value=(7, 7)         # Change the Gaussian blur values (must be odd numbers)
   )
   ```

3. **Adjust Parameters (Optional)**:

   - `method`: Choose the feature detection method to use for image alignment. Default is `'ORB'`.
   - `align`: Set to `True` to align images before comparison. Useful for images taken from different angles or positions.
   - `sensitivity_threshold`: Controls the sensitivity to differences between the images. Lower values highlight more subtle differences.
   - `blur_value`: Tuple representing the Gaussian blur kernel size. Larger values increase blur effect, highlighting significant differences and ignoring minor ones.

4. **View the Output**: The function saves an image named 'comparison\_{method}.jpg' in the 'output/' directory with the differences highlighted. You can comment/uncomment the `cv2.imshow()` line in the function to display the image directly.

5. **Experiment**: Adjust `method`, `sensitivity_threshold`, `blur_value`, and `align` settings to fine-tune the comparison for your specific images.

Ensure all dependencies are installed as per the 'Installing' section before using PixCompare.

## Testing

PixCompare includes comprehensive unit tests to ensure its functionality and reliability. To run the tests, execute the following command from the project directory:

```bash
python test_pix_compare.py
```

The test suite covers various scenarios, including parameter validation, method testing with alignment options, and image loading/saving. The test results, including execution times for each test, will be displayed in the console.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is open source and available under the [MIT License](LICENSE).
