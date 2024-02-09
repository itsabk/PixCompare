import cv2
import numpy as np

def highlight_unique_features(image_path1, image_path2, sensitivity_threshold=45, blur_value=(5, 5)):
    """
    Highlights unique features between two images.

    Args:
    image_path1 (str): Path to the first image.
    image_path2 (str): Path to the second image.
    sensitivity_threshold (int, optional): Threshold for highlighting differences. Defaults to 45.
    blur_value (tuple, optional): The size of the Gaussian blur kernel. Defaults to (5, 5).

    This function loads two images, applies Gaussian blur, and then highlights the differences
    between them. The differences are shown in color against a grayscale background.
    """

    # Load the images from the given file paths
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    
    # Adjust blur_value to ensure it contains odd integers
    blur_value = (max(1, blur_value[0] // 2 * 2 + 1), max(1, blur_value[1] // 2 * 2 + 1))

    # Resize the second image to match the dimensions of the first
    h, w = image1.shape[:2]
    image2 = cv2.resize(image2, (w, h))

    # Convert both images to grayscale for processing
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to both grayscale images
    blurred1 = cv2.GaussianBlur(gray1, blur_value, 0)
    blurred2 = cv2.GaussianBlur(gray2, blur_value, 0)

    # Calculate the differences between the blurred images
    diff1 = cv2.subtract(blurred1, blurred2)  # Present in image1 but not in image2
    diff2 = cv2.subtract(blurred2, blurred1)  # Present in image2 but not in image1

    # Apply a binary threshold to highlight significant differences
    _, thresh_diff1 = cv2.threshold(diff1, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    _, thresh_diff2 = cv2.threshold(diff2, sensitivity_threshold, 255, cv2.THRESH_BINARY)

    # Highlight differences in color (blue for image1, red for image2)
    highlighted1 = cv2.merge([thresh_diff1, thresh_diff1, np.zeros_like(thresh_diff1)])  # Blue-Green for unique in image1
    highlighted2 = cv2.merge([np.zeros_like(thresh_diff2), np.zeros_like(thresh_diff2), thresh_diff2])  # Red for unique in image2

    # Combine the highlighted images
    highlighted = cv2.addWeighted(highlighted1, 1, highlighted2, 1, 0)

    # Convert the first grayscale image to BGR for background
    gray_image = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)

    # Overlay the color highlighted differences onto the grayscale background
    final_image = cv2.addWeighted(gray_image, 1, highlighted, 1, 0)

    # Save the final image with highlighted differences
    cv2.imwrite('highlighted_output.jpg', final_image)

    # Uncomment below lines to display the image
    # cv2.imshow('Unique Highlighted Difference', final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage of the function
highlight_unique_features('image1.jpg', 'image2.jpg')
