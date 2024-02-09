# PixCompare

This project provides a Python tool to highlight unique features between two images. It's useful for comparing images to identify differences visually.

## Features

- Load two images and resize them to the same dimensions.
- Apply Gaussian blur to both images.
- Highlight differences in color on a grayscale background.

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

To use PixCompare, follow these steps:

1. Prepare your images: Ensure you have two images that you want to compare. These images can be of different sizes.

2. Use the `highlight_unique_features` function: This function is the core of PixCompare. Here's how to use it:

   ```python
   from pix_compare import highlight_unique_features

   # Basic usage with default settings
   highlight_unique_features('path/to/image1.jpg', 'path/to/image2.jpg')

   # Advanced usage with custom settings
   highlight_unique_features(
       'path/to/image1.jpg',
       'path/to/image2.jpg',
       sensitivity_threshold=50,  # Adjust the sensitivity of the difference highlighting
       blur_value=(7, 7)          # Change the Gaussian blur values (must be odd numbers)
   )
   ```

3. Adjust Parameters (Optional):

   - `sensitivity_threshold`: This parameter controls how sensitive the tool is to differences between the images. A lower value will highlight more subtle differences, while a higher value will only highlight more significant differences.

   - `blur_value`: This tuple represents the size of the Gaussian blur kernel. Larger values increase the blur effect, which can help in highlighting larger, more significant differences while ignoring minor ones. The values must be odd integers.

4. Check the output: The function saves the resulting image with highlighted differences as 'highlighted_output.jpg' in your working directory. You can also uncomment the lines at the end of the function to display the image directly.

5. Experiment with different settings: Feel free to adjust the `sensitivity_threshold` and `blur_value` parameters to fine-tune the difference highlighting according to your specific images.

Remember to install the required dependencies as described in the 'Installing' section before using PixCompare.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is open source and available under the [MIT License](LICENSE).
