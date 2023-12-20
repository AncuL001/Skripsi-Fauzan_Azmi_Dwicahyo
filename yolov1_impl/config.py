from dataclasses import dataclass, replace
import torch

@dataclass(frozen=True)
class Config:
    LEARNING_RATE: float = 2e-5
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE: int = 64
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 100
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    LOAD_MODEL: bool = False
    LOAD_MODEL_FILE: str = "overfit.pth.tar"

    BASE_SAVE_LOG_PATH: str = "../runs"

    DROPOUT: float = 0

    DATASET_PATH: str = '../downloads/TACO/data'
    anns_file_path: str = DATASET_PATH + '/' + 'annotations.json'

    VAL_DATASET_PATH: str = '../downloads/TACO/data'
    val_anns_file_path: str = DATASET_PATH + '/' + 'annotations.json'

    TEST_DATASET_PATH: str = '../downloads/TACO/data'
    test_anns_file_path: str = DATASET_PATH + '/' + 'annotations.json'

    train_indices_path: str = 'train_indices-2023-17-11_13-41-46.csv'
    test_indices_path: str = 'test_indices-2023-17-11_13-41-46.csv'

    IMAGE_SIZE: int = 448
    scale: float = 1.12

    threshold: float = 0.4
    iou_threshold: float = 0.5
    box_format: str = "midpoint"
    SPLIT_SIZE: int = 7
    NUM_BOXES: int = 2
    NUM_CLASSES: int = 1

    multiples_to_log_train_map: int = 5

    def replace(self, **kwargs):
        return replace(self, **kwargs)
