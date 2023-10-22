import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import YoloV1
from taco_dataset import CoCoDatasetForYOLO
from loss import YoloLoss
from utils import (
    mean_average_precision,
    get_bboxes,
    cellboxes_to_boxes,
    non_max_suppression,
)
import datetime

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"

DATASET_PATH = '../downloads/TACO/data'
anns_file_path = DATASET_PATH + '/' + 'annotations.json'

IMAGE_SIZE = 448
scale = 1.12

threshold=0.4
iou_threshold = 0.5
box_format="midpoint"
SPLIT_SIZE=7
NUM_BOXES=2
NUM_CLASSES=1

multiples_to_log_train_map = 5

train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    
    # got this error
    # Expected y_min for bbox (0.5355392156862745, -0.00015318627450980338, 0.6523692810457516, 0.1803002450980392, 0) to be in the range [0.0, 1.0], got -0.00015318627450980338.
    # rounding issue :/
    # the insane solution to this problem (modifying the library code) https://github.com/albumentations-team/albumentations/issues/459#issuecomment-734454278
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

def main():
    writer = SummaryWriter()

    model = YoloV1(split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss(S=SPLIT_SIZE, B=NUM_BOXES, C=NUM_CLASSES)

    train_dataset = CoCoDatasetForYOLO(
        root=DATASET_PATH,
        annFile=anns_file_path,
        transform=train_transforms,
        C=1
    )

    test_dataset = CoCoDatasetForYOLO(
        root=DATASET_PATH,
        annFile=anns_file_path,
        transform=test_transforms,
        C=1
    )

    # # for testing with a small dataset
    # training_portion = list(range(0, 32))
    # testing_portion = list(range(32, 64))
    # train_dataset = torch.utils.data.Subset(dataset, training_portion)
    # test_dataset = torch.utils.data.Subset(dataset, testing_portion)

    train_percentage = 0.8

    indices = torch.randperm(len(train_dataset))
    test_size = round(len(train_dataset) * (1 - train_percentage))

    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-test_size])
    test_dataset = torch.utils.data.Subset(test_dataset, indices[-test_size:])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, leave=True)
        train_losses = []

        model.train()
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)

            loss = loss_fn(out, y)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        writer.add_scalar('loss/train', sum(train_losses)/len(train_losses), epoch)

        model.eval()
        with torch.no_grad():
            test_losses = []

            test_pred_boxes = []
            test_target_boxes = []

            train_idx = 0

            for batch_idx, (x, labels) in enumerate(test_loader):
                x, labels = x.to(DEVICE), labels.to(DEVICE)
                predictions = model(x)

                loss = loss_fn(predictions, y)
                test_losses.append(loss.item())

                batch_size = x.shape[0]
                true_bboxes = cellboxes_to_boxes(labels, S=SPLIT_SIZE, B=NUM_BOXES, C=NUM_CLASSES)
                bboxes = cellboxes_to_boxes(predictions, S=SPLIT_SIZE, B=NUM_BOXES, C=NUM_CLASSES)

                for idx in range(batch_size):
                    nms_boxes = non_max_suppression(
                        bboxes[idx],
                        iou_threshold=iou_threshold,
                        threshold=threshold,
                        box_format=box_format,
                    )

                    for nms_box in nms_boxes:
                        test_pred_boxes.append([train_idx] + nms_box)

                    for box in true_bboxes[idx]:
                        # many will get converted to 0 pred
                        if box[1] > threshold:
                            test_target_boxes.append([train_idx] + box)

                    train_idx += 1

            test_mean_avg_prec = mean_average_precision(
                test_pred_boxes, test_target_boxes, iou_threshold=iou_threshold, box_format=box_format
            )

            writer.add_scalar('loss/test', sum(test_losses)/len(test_losses), epoch)
            writer.add_scalar('mAP/test', test_mean_avg_prec, epoch)

        if epoch % multiples_to_log_train_map != 0:
            continue

        with torch.no_grad():
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=iou_threshold, threshold=threshold, device=DEVICE,
                S=SPLIT_SIZE, B=NUM_BOXES, C=NUM_CLASSES
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=iou_threshold, box_format=box_format
            )

            writer.add_scalar('mAP/train', mean_avg_prec, epoch)

    writer.close()

    model_scripted = torch.jit.script(model)
    model_scripted.save('../downloads/yolo_v1_model.pt')


if __name__ == "__main__":
    print(f"Run started on {datetime.datetime.now()}")
    main()
    print(f"Run ended on {datetime.datetime.now()}")