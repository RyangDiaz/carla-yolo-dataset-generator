import imageio
from PIL import Image
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model.predict('carla_real2.mp4', stream=True, device=1)
predict_frames = [r.plot()[..., ::-1] for r in results]
f = 'yolo_real_predict2.mp4'
imageio.mimwrite(f, predict_frames, fps=20)
print(f'Saved output prediction video to {f}')