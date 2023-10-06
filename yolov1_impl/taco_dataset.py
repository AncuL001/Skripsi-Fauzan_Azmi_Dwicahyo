"""
Creates a Pytorch dataset to load the TACO dataset
"""

import torch
from torchvision.datasets import CocoDetection
from utils import (
    get_correctly_rotated_image
)

class CoCoDatasetForYOLO(CocoDetection):
    """
    a

    Args:
        S: grid size
        B: number of bounding boxes per grid cell
        C: number of classes

        (inherited from CoCoDetection)
        root: root directory of dataset
        annFile: path to json annotation file
        transform: transform to apply to the image
    """
    def __init__(
        self, S=7, B=2, C=0, **kwargs
    ):
        super().__init__(**kwargs)
        self.S = S
        self.B = B
        self.C = C

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        image = get_correctly_rotated_image(image)

        img_width, img_height = image.size
        boxes = []

        for instance in target:
            x, y, width, height = instance['bbox']
            boxes.append([0, x/img_width, y/img_height, width/img_width, height/img_height])

        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:

            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)

            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates

                # Set one hot encoding for class_label
                # We only have one class here, which is just litter
                label_matrix[i, j, 0] = 1

        return image, label_matrix