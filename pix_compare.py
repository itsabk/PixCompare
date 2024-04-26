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
        tuple: A tuple containing the following elements:
            - aligned_image (numpy.ndarray): The aligned version of the first image.
            - homography (numpy.ndarray): The homography matrix used for the alignment.
            - success (bool): A boolean indicating whether the alignment was successful.
            - crop_coords (tuple): A tuple containing the cropping coordinates (x_min, x_max, y_min, y_max) if auto_crop is True, otherwise None.
    """
    im1_gray, im2_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = detector.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = detector.detectAndCompute(im2_gray, None)
    
    matcher = cv2.BFMatcher(cv2.NORM_L2 if descriptors1.dtype == np.float32 else cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        height, width = im2.shape[:2]
        im1_aligned = cv2.warpPerspective(im1, h, (width, height))
        
        if auto_crop:
            mask = cv2.cvtColor(im1_aligned, cv2.COLOR_BGR2GRAY) > 0
            coords = np.argwhere(mask)
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

    Returns:
        numpy.ndarray: The final image with highlighted differences.
    """
    blur_value = (max(1, blur_value[0] // 2 * 2 + 1), max(1, blur_value[1] // 2 * 2 + 1))
    gray1, gray2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cv2.resize(image2, image1.shape[:2][::-1]), cv2.COLOR_BGR2GRAY)
    blurred1, blurred2 = cv2.GaussianBlur(gray1, blur_value, 0), cv2.GaussianBlur(gray2, blur_value, 0)
    
    diff1, diff2 = cv2.subtract(blurred1, blurred2), cv2.subtract(blurred2, blurred1)
    _, thresh_diff1 = cv2.threshold(diff1, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    _, thresh_diff2 = cv2.threshold(diff2, sensitivity_threshold, 255, cv2.THRESH_BINARY)
    
    highlighted = cv2.addWeighted(
        cv2.merge([thresh_diff1, thresh_diff1, np.zeros_like(thresh_diff1)]), 1, 
        cv2.merge([np.zeros_like(thresh_diff2), np.zeros_like(thresh_diff2), thresh_diff2]), 1, 0)
    
    return cv2.addWeighted(cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR), 1, highlighted, 1, 0)

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

    Returns:
        None
    """
    try:
        image1, image2 = cv2.imread(image_path1), cv2.imread(image_path2)
        if image1 is None or image2 is None:
            raise ValueError("Failed to load one or both image files.")
        
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
        
        detector = method_dict.get(method)
        if detector is None:
            raise ValueError("Invalid feature detection method. Please choose from: SIFT, BRISK, AKAZE, KAZE, BRIEF, FREAK, LATCH, LUCID, DAISY, or ORB.")
        
        if align:
            image1_aligned, _, alignment_success, crop_coords = align_images(image1, image2, detector=detector, auto_crop=auto_crop)
            if alignment_success:
                image1 = image1_aligned
        
        highlighted_image = highlight_differences(image1, image2, sensitivity_threshold, blur_value)
        
        if auto_crop and crop_coords:
            x_min, x_max, y_min, y_max = crop_coords
            highlighted_image = highlighted_image[x_min:x_max+1, y_min:y_max+1]
        
        # Save the output image with method name in the filename
        cv2.imwrite(f'output/comparison_{method}.jpg', highlighted_image)
    
    except IOError as e:
        print(f"Error: {str(e)}")
    except ValueError as e:
        raise e

# Example usage
compare_images('image1.jpg', 'image2.jpg', method='SIFT', align=True, auto_crop=True, sensitivity_threshold=40, blur_value=(7, 7))