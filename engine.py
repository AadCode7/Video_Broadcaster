import cv2
from ultralytics import YOLO
import numpy as np
import torch

class CustomSegmentationWithYolo():
    def __init__(self, erode_size=5, erode_intensity=2):
        self.model = YOLO('yolov8m-seg.pt')
        self.erode_size = erode_size
        self.erode_intensity = erode_intensity
        self.background_image = cv2.imread("./static/office.jpg")

    def generate_mask_from_result(self, results):
        for result in results:
            if result.masks is not None and len(result.masks) > 0:
                try:
                    masks = result.masks.data
                    boxes = result.boxes.data
                
                    if boxes.shape[0] == 0:  # No detections
                        return None
                
                # Check if class column exists
                    if boxes.shape[1] > 5:
                        clss = boxes[:, 5]
                        people_indices = torch.where(clss==0)
                        people_masks = masks[people_indices]
                    else:
                        # If no class info, assume all detections are people
                        people_masks = masks
                
                    if people_masks.numel() == 0:
                        return None
                
                    people_mask = torch.any(people_masks, dim=0).to(torch.uint8) * 255
                
                # Convert to numpy if not already
                    if isinstance(people_mask, torch.Tensor):
                        people_mask = people_mask.cpu().numpy()
                
                    kernel = np.ones((self.erode_size, self.erode_size), np.uint8)
                    eroded_mask = cv2.erode(people_mask, kernel, iterations=self.erode_intensity)
                
                    return eroded_mask
                except Exception as e:
                    print(f"Error generating mask: {e}")
                    return None
            else:
                return None
    
        return None
            
    def apply_blur_with_mask(self, frame, mask, blur=21):
        blur = (blur, blur)
        blurred_frame = cv2.GaussianBlur(frame, blur,0)

        mask = (mask > 0).astype(np.uint8)
        mask_3d = cv2.merge([mask, mask, mask])

        
        result_frame = np.where(mask_3d ==1, frame, blurred_frame)

        return result_frame
    
    def apply_black_background(self, frame, mask):
        black_background = np.zeros_like(frame)
        result_frame = np.where(mask[:,:,np.newaxis] == 255, frame, black_background)
        return result_frame
    
    def apply_custom_background(self,frame,mask):
        background_image = cv2.resize(self.background_image,(frame.shape[1], frame.shape[0]))
        result_frame = np.where(mask[:,:,np.newaxis] == 255, frame, background_image)
        return result_frame