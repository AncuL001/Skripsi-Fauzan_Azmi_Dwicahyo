import cv2
import config
from models.resnet34_yolo import resnet34_yolo as used_model
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import (
  cellboxes_to_boxes,
  non_max_suppression,
)
import numpy as np

def plot_image(image, boxes):
  """Plots predicted bounding boxes on the image"""
  im = np.array(image)
  height, width, _ = im.shape

  # Display the image

  # box[0] is x midpoint, box[2] is width
  # box[1] is y midpoint, box[3] is height

  # Create a Rectangle potch
  for box in boxes:
    box = box[2:]
    assert len(box) == 4, "Got more values than x, y, w, h, in a box!"
    upper_left_x = box[0] - box[2] / 2
    upper_left_y = box[1] - box[3] / 2
    bottom_right_x = box[0] + box[2] / 2
    bottom_right_y = box[1] + box[3] / 2
    cv2.rectangle(
      image,
      (int(upper_left_x * width), int(upper_left_y * height)),
      (int(bottom_right_x * width), int(bottom_right_y * height)),
      (0, 0, 255),
      2
    )

  cv2.imshow("image", image)

def main():
  camera = cv2.VideoCapture(0)

  folder = "../runs/ResNet/dropout-0.1/2023-02-12_14-14-14"
  path_to_model = folder + "/model.pt"
  dropout = 0.1

  cfg = config.Config().replace(DROPOUT=0.1, BATCH_SIZE=8)
  model = used_model(
    split_size=cfg.SPLIT_SIZE, 
    num_boxes=cfg.NUM_BOXES, 
    num_classes=cfg.NUM_CLASSES, 
    dropout_percentage=dropout).to(cfg.DEVICE)

  model.load_state_dict(
    torch.load(path_to_model, map_location=torch.device(cfg.DEVICE))
  )
  model.eval()

  IMAGE_SIZE = 448

  data_preprocess = A.Compose(
    [
      A.LongestMaxSize(max_size=IMAGE_SIZE),
      A.PadIfNeeded(
        min_height=IMAGE_SIZE, 
        min_width=IMAGE_SIZE, 
        border_mode=cv2.BORDER_CONSTANT
      ),
      A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
      ToTensorV2(),
    ],
  )

  while True:
    ret, image = camera.read()

    cvt_img = data_preprocess(image=image)["image"].unsqueeze(0)
    pred = model(cvt_img)
    pred = cellboxes_to_boxes(pred, S=7, B=2, C=1)
    pred = non_max_suppression(
      pred[0], 
      iou_threshold=0.5, 
      threshold=0.4, 
      box_format="midpoint"
    )

    plot_image(image, pred)
    if (cv2.waitKey(50) == 27):
      break

  cv2.destroyAllWindows()
  camera.release()

if (__name__ == "__main__"):
  main()