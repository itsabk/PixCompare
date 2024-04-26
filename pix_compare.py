import cv2
import numpy as np

def align_images(im1, im2, detector, auto_crop=False):
    """
    Aligns two images using feature detection and homography.

    Args:
    im1 (numpy.ndarray): The first image to be aligned.
    im2 (numpy.ndarray): The second image to which the first image is aligned.
    detector: The feature detection method to use.
    auto_crop (bool, optional): Whether to automatically crop the black borders introduced during alignment. Defaults to False.

    Returns:
    tuple: A tuple containing the aligned image, the homography matrix, a boolean indicating success, and the cropping coordinates (if auto_crop is True).
    """
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = detector.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = detector.detectAndCompute(im2_gray, None)

    # Create matcher based on descriptor type
    if descriptors1.dtype == np.float32:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to find good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Minimum number of good matches to proceed
    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        # Extract locations of matched keypoints
        src_pts = np.float32(
            [keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Use homography to warp image
        height, width, channels = im2.shape
        im1_aligned = cv2.warpPerspective(im1, h, (width, height))

        if auto_crop:
            # Create a mask of non-black pixels
            mask = cv2.cvtColor(im1_aligned, cv2.COLOR_BGR2GRAY) > 0

            # Find the coordinates of non-black pixels
            coords = np.argwhere(mask)

            # Find the minimum and maximum coordinates
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)

            return im1_aligned, h, True, (x_min, x_max, y_min, y_max)
        else:
            return im1_aligned, h, True, None
    else:
        print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
        return im1, None, False, None

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

    return final_image

def compare_images(image_path1, image_path2, method='ORB', align=False, auto_crop=False, sensitivity_threshold=45, blur_value=(21, 21)):
    """
    Processes two images by optionally aligning them and highlighting unique features.

    Args:
    image_path1 (str): Path to the first image.
    image_path2 (str): Path to the second image.
    method (str, optional): The feature detection method to use. Defaults to 'ORB'.
    align (bool, optional): Whether to align the images before highlighting differences. Defaults to False.
    auto_crop (bool, optional): Whether to automatically crop the black borders introduced during alignment. Defaults to False.
    sensitivity_threshold (int, optional): Threshold for highlighting differences. Defaults to 45.
    blur_value (tuple, optional): The size of the Gaussian blur kernel. Defaults to (21, 21).
    """
    try:
        # Load the images from the given file paths
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)

        if image1 is None or image2 is None:
            raise ValueError("Failed to load one or both image files.")

        # Dictionary mapping method names to their respective OpenCV function calls
        method_dict = {
            'SIFT': cv2.SIFT_create(),
            'BRISK': cv2.BRISK_create(),
            'AKAZE': cv2.AKAZE_create(),
            'KAZE': cv2.KAZE_create(),
            'BRIEF': cv2.xfeatures2d.BriefDescriptorExtractor_create(),
            'FREAK': cv2.xfeatures2d.FREAK_create(),
            'LATCH': cv2.xfeatures2d.LATCH_create(),
            'LUCID': cv2.xfeatures2d.LUCID_create(),
            'DAISY': cv2.xfeatures2d.DAISY_create(),
            'ORB': cv2.ORB_create(5000)
        }

        # Get the selected method from the dictionary
        if method in method_dict:
            detector = method_dict[method]
        else:
            raise ValueError("Invalid feature detection method. Please choose from: SIFT, BRISK, AKAZE, KAZE, BRIEF, FREAK, LATCH, LUCID, DAISY, or ORB.")

        # Align images if required
        if align:
            image1_aligned, _, alignment_success, crop_coords = align_images(image1, image2, detector=detector, auto_crop=auto_crop)
            if alignment_success:
                image1 = image1_aligned

        # Highlight the differences
        highlighted_image = highlight_differences(image1, image2, sensitivity_threshold, blur_value)

        # Crop the highlighted image if auto_crop is True and crop_coords are available
        if auto_crop and crop_coords:
            x_min, x_max, y_min, y_max = crop_coords
            highlighted_image = highlighted_image[x_min:x_max+1, y_min:y_max+1]

        # Save the final image with highlighted differences
        cv2.imwrite('highlighted_output.jpg', highlighted_image)

    except (IOError, ValueError) as e:
        print(f"Error: {str(e)}")

# Example usage
compare_images('image1.jpg', 'image2.jpg', method='SIFT', align=True, auto_crop=True, sensitivity_threshold=40, blur_value=(7, 7))