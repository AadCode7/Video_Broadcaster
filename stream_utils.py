import cv2
import pyvirtualcam
import torch
import threading
from ultralytics import YOLO

from engine import CustomSegmentationWithYolo

class Streaming(CustomSegmentationWithYolo):
    def __init__(self, in_source=None, out_source=None, fps=None, blur=None, cam_fps=30, background="none", preview=False):
        super().__init__(erode_size=5, erode_intensity=2)
        
        self.input_source = in_source
        self.output_source = out_source
        self.fps = fps
        self.blur = blur
        self.preview = preview
        self.background = background
        self.original_fps = cam_fps
        self.running = False
        self.model = YOLO('yolov8m-seg.pt').to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def list_available_devices(self):
        devices = []
        for i in range(2):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append({"id": i, "name": f"Camera {i}"})
                cap.release()
        return devices
    
    def stream_video(self):
        self.running = True
        cap = cv2.VideoCapture(int(self.input_source))

        frame_idx = 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            self.original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        except Exception as e:
            print(f"Webcam {self.input_source} live FPS not available, Change FPS accordingly. Exception: {e}")
        
        if self.fps:
            if self.fps > self.original_fps:
                self.fps = self.original_fps
            frame_interval = max(1, int(self.original_fps / self.fps))
        else:
            frame_interval = 1

        with pyvirtualcam.Camera(width=width, height=height, fps=self.fps) as cam:
            print(f"Virtual Camera running at {width}x{height} at {self.fps} fps.")

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                result_frame = frame  # Default to original frame
                
                if frame_idx % frame_interval == 0:
                    results = self.model.predict(source=frame, save=False, save_txt=False, stream=True, retina_masks=True, verbose=False)
                    mask = self.generate_mask_from_result(results)
                    
                    if mask is not None:
                        if self.background == "blur":
                            result_frame = self.apply_blur_with_mask(frame, mask, blur=self.blur) 
                        elif self.background == "none":
                            result_frame = self.apply_black_background(frame, mask)
                        elif self.background == "default":
                            result_frame = self.apply_custom_background(frame, mask)

                if self.preview:
                    cv2.imshow('Preview', result_frame)

                cam.send(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()
                frame_idx += 1

                if cv2.waitKey(1) == 27:
                    self.running = False
                    break
            if self.preview:
                cv2.destroyAllWindows()

        cap.release()

    def update_streaming_config(self, in_source=None, out_source=None, fps=None, blur=None, background="none"):
        if in_source is not None:
            self.input_source = in_source
        if out_source is not None:
            self.output_source = out_source
        if fps is not None:
            self.fps = fps
        if blur is not None:
            self.blur = blur
        if background is not None:
            self.background = background

    def update_running_status(self, running_status=False):
        self.running = running_status
        return {"message": "Stream stopped successfully"}