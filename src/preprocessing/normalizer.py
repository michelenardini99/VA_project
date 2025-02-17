import cv2
import numpy as np

def normalize_roi(image, landmarks):
    
    def get_affine_transform(src_points, dst_points):
        
        return cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))

    def apply_offsets(landmarks, offset):
        
        return [(x + offset[0], y + offset[1]) for (x, y) in landmarks]

    def warp_roi(image, transform_matrix, size):
        
        return cv2.warpAffine(image, transform_matrix, size, flags=cv2.INTER_LINEAR)

    
    reference_landmarks = {
        'face': [(30, 40), (100, 40), (65, 128)],  
        'left_eye': [(26, 40), (10, 10), (42, 10)],  
        'right_eye': [(26, 40), (10, 10), (42, 10)],  
        'mouth': [(10, 20), (46, 20), (28, 50)]  
    }

    target_sizes = {
        'face': (150, 130),
        'left_eye': (52, 52),
        'right_eye': (52, 52),
        'mouth': (56, 62)
    }


    normalized_rois = {}

    for roi_key, landmarks in landmarks.items():
        if roi_key in reference_landmarks:
            
            ref_points = reference_landmarks[roi_key]

            target_size = target_sizes[roi_key]
            
            transform_matrix = cv2.getAffineTransform(np.float32(landmarks[:3]), np.float32(ref_points))

            normalized_roi = cv2.warpAffine(image, transform_matrix, target_size, flags=cv2.INTER_LINEAR)
            normalized_rois[roi_key] = normalized_roi

    return normalized_rois
