# PixCompare

PixCompare is a Python tool designed to highlight unique features between two images, making it easier to visually identify differences. This tool is particularly useful for comparing images that may not be perfectly aligned or are of different sizes.

## Features

- **Image Alignment**: Aligns two images using ORB feature detection and homography before comparing them.
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

- Clone the repository:
  ```bash
  git clone https://github.com/itsabk/PixCompare.git
  ```
- Install required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Usage

To utilize PixCompare, follow these steps:

1. **Prepare Your Images**: Ensure you have two images that you wish to compare.

2. **Use the `compare_images` Function**: This function is central to PixCompare. It offers options for aligning images before highlighting differences.

   ```python
   from pix_compare import compare_images

   # Basic usage with image alignment
   compare_images('path/to/image1.jpg', 'path/to/image2.jpg', align=True)

   # Advanced usage with custom settings for sensitivity and blur
   compare_images(
       'path/to/image1.jpg',
       'path/to/image2.jpg',
       align=True,
       sensitivity_threshold=40,  # Adjust the sensitivity of difference highlighting
       blur_value=(7, 7)          # Change the Gaussian blur values (must be odd numbers)
   )
   ```

3. **Adjust Parameters (Optional)**:

   - `align`: Set to `True` to align images before comparison. Useful for images taken from different angles or positions.
   - `sensitivity_threshold`: Controls the sensitivity to differences between the images. Lower values highlight more subtle differences.
   - `blur_value`: Tuple representing the Gaussian blur kernel size. Larger values increase blur effect, highlighting significant differences and ignoring minor ones.

4. **View the Output**: The function saves an image named 'highlighted_output.jpg' in your working directory with the differences highlighted. You can uncomment lines in the function to display the image directly.

5. **Experiment**: Adjust `sensitivity_threshold`, `blur_value`, and `align` settings to fine-tune the comparison for your specific images.

Ensure all dependencies are installed as per the 'Installing' section before using PixCompare.

## Testing

PixCompare includes comprehensive unit tests to ensure its functionality and reliability. To run the tests, execute the following command:

```bash
python test_pix_compare.py
```

The test suite covers various scenarios, including parameter validation and method testing with alignment options.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is open source and available under the [MIT License](LICENSE).
