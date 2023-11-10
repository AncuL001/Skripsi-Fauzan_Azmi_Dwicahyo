import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.yolo import YoloV1
from taco_dataset import CoCoDatasetForYOLO
from loss import YoloLoss
from utils import (
    mean_average_precision,
    get_bboxes,
    cellboxes_to_boxes,
    non_max_suppression,
    write_to_file,
)
import datetime
from config import Config
from json import dumps
from torchinfo import summary

def train_loop(model, train_dataset, test_dataset, optimizer, loss_fn, writer, cfg: Config):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(cfg.EPOCHS):
        loop = tqdm(train_loader, leave=True)
        train_losses = []

        model.train()
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
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
                x, labels = x.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                predictions = model(x)

                loss = loss_fn(predictions, y)
                test_losses.append(loss.item())

                batch_size = x.shape[0]
                true_bboxes = cellboxes_to_boxes(labels, S=cfg.SPLIT_SIZE, B=cfg.NUM_BOXES, C=cfg.NUM_CLASSES)
                bboxes = cellboxes_to_boxes(predictions, S=cfg.SPLIT_SIZE, B=cfg.NUM_BOXES, C=cfg.NUM_CLASSES)

                for idx in range(batch_size):
                    nms_boxes = non_max_suppression(
                        bboxes[idx],
                        iou_threshold=cfg.iou_threshold,
                        threshold=cfg.threshold,
                        box_format=cfg.box_format,
                    )

                    for nms_box in nms_boxes:
                        test_pred_boxes.append([train_idx] + nms_box)

                    for box in true_bboxes[idx]:
                        # many will get converted to 0 pred
                        if box[1] > cfg.threshold:
                            test_target_boxes.append([train_idx] + box)

                    train_idx += 1

            test_mean_avg_prec = mean_average_precision(
                test_pred_boxes, test_target_boxes, iou_threshold=cfg.iou_threshold, box_format=cfg.box_format
            )

            writer.add_scalar('loss/test', sum(test_losses)/len(test_losses), epoch)
            writer.add_scalar('mAP/test', test_mean_avg_prec, epoch)

        if epoch % cfg.multiples_to_log_train_map != 0:
            continue

        with torch.no_grad():
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=cfg.iou_threshold, threshold=cfg.threshold, device=cfg.DEVICE,
                S=cfg.SPLIT_SIZE, B=cfg.NUM_BOXES, C=cfg.NUM_CLASSES
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=cfg.iou_threshold, box_format=cfg.box_format
            )

            writer.add_scalar('mAP/train', mean_avg_prec, epoch)

    writer.close()

def main(cfg: Config):
    model = YoloV1(split_size=cfg.SPLIT_SIZE, num_boxes=cfg.NUM_BOXES, num_classes=cfg.NUM_CLASSES, dropout_percentage=cfg.DROPOUT).to(cfg.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    loss_fn = YoloLoss(S=cfg.SPLIT_SIZE, B=cfg.NUM_BOXES, C=cfg.NUM_CLASSES)

    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=int(cfg.IMAGE_SIZE * cfg.scale)),
            A.PadIfNeeded(
                min_height=int(cfg.IMAGE_SIZE * cfg.scale),
                min_width=int(cfg.IMAGE_SIZE * cfg.scale),
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.RandomCrop(width=cfg.IMAGE_SIZE, height=cfg.IMAGE_SIZE),
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
            A.LongestMaxSize(max_size=cfg.IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=cfg.IMAGE_SIZE, min_width=cfg.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    )

    train_dataset = CoCoDatasetForYOLO(
        root=cfg.DATASET_PATH,
        annFile=cfg.anns_file_path,
        transform=train_transforms,
        S=cfg.SPLIT_SIZE, B=cfg.NUM_BOXES, C=cfg.NUM_CLASSES
    )

    test_dataset = CoCoDatasetForYOLO(
        root=cfg.DATASET_PATH,
        annFile=cfg.anns_file_path,
        transform=test_transforms,
        S=cfg.SPLIT_SIZE, B=cfg.NUM_BOXES, C=cfg.NUM_CLASSES
    )

    # # for testing with a small dataset
    # training_portion = list(range(0, 32))
    # testing_portion = list(range(32, 64))
    # train_dataset = torch.utils.data.Subset(train_dataset, training_portion)
    # test_dataset = torch.utils.data.Subset(test_dataset, testing_portion)

    train_percentage = 0.8

    indices = torch.randperm(len(train_dataset))
    test_size = round(len(train_dataset) * (1 - train_percentage))

    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-test_size])
    test_dataset = torch.utils.data.Subset(test_dataset, indices[-test_size:])

    full_log_folder = f"{cfg.BASE_SAVE_LOG_PATH}/{model._get_name()}/dropout-{cfg.DROPOUT}/{datetime.datetime.now().strftime('%Y-%d-%m_%H-%M-%S')}"

    writer = SummaryWriter(log_dir=full_log_folder)
    write_to_file(path=f"{full_log_folder}/config.txt", text=dumps(config.__dict__, indent=2))
    write_to_file(path=f"{full_log_folder}/model_summary.txt", text=str(summary(model, input_size=(1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))))
    write_to_file(path=f"{full_log_folder}/model_structure.txt", text=model.__str__())

    train_loop(model, train_dataset, test_dataset, optimizer, loss_fn, writer, cfg)

    torch.save(model.state_dict(), filename=f"{full_log_folder}/model.pt")


if __name__ == "__main__":
    config = Config()
    main(cfg=config)