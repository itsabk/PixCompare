import cv2
import numpy as np

def align_images(im1, im2):
    """
    Aligns two images using ORB feature detection and homography.

    Args:
    im1 (numpy.ndarray): The first image to be aligned.
    im2 (numpy.ndarray): The second image to which the first image is aligned.

    Returns:
    tuple: A tuple containing the aligned image, the homography matrix, and a boolean indicating success.
    """
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(5000)  # increased key points

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to find good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Minimum number of good matches to proceed
    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        # Extract locations of matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Use homography to warp image
        height, width, channels = im2.shape
        im1_aligned = cv2.warpPerspective(im1, h, (width, height))
        return im1_aligned, h, True
    else:
        print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
        return im1, None, False

def highlight_differences(image1, image2, sensitivity_threshold=45, blur_value=(21, 21)):
    """
    Highlights unique features between two images.

    Args:
    image1 (numpy.ndarray): The first image.
    image2 (numpy.ndarray): The second image.
    sensitivity_threshold (int, optional): Threshold for highlighting differences. Defaults to 45.
    blur_value (tuple, optional): The size of the Gaussian blur kernel. Defaults to (21, 21).

    This function takes two images, applies Gaussian blur, and then highlights the differences
    between them. The differences are shown in color against a grayscale background.
    """
    
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

    return 'highlighted_output.jpg'

def compare_images(image_path1, image_path2, align=False, sensitivity_threshold=45, blur_value=(21, 21)):
    """
    Processes two images by optionally aligning them and highlighting unique features.

    Args:
    image_path1 (str): Path to the first image.
    image_path2 (str): Path to the second image.
    align (bool, optional): Whether to align the images before highlighting differences. Defaults to False.
    sensitivity_threshold (int, optional): Threshold for highlighting differences. Defaults to 45.
    blur_value (tuple, optional): The size of the Gaussian blur kernel. Defaults to (21, 21).
    """
    # Load the images from the given file paths
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Align images if required
    if align:
        image1_aligned, _, alignment_success = align_images(image1, image2)
        if alignment_success:
            image1 = image1_aligned

    # Highlight the differences
    highlight_differences(image1, image2, sensitivity_threshold, blur_value)

# Example usage
compare_images('image1.jpg', 'image2.jpg', align=True, sensitivity_threshold=40, blur_value=(7, 7))
