import os
import numpy as np
import torch
from PIL import Image, ImageDraw

class AppleDataset(object):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, "masks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_img_name(self, idx):
        return self.imgs[idx]

    def save_image_with_boxes(self, idx, save_dir):
        img, target = self.__getitem__(idx)
        
        if isinstance(img, torch.Tensor):
            img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

        draw = ImageDraw.Draw(img)
        for box in target["boxes"]:
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img.save(os.path.join(save_dir, f"img_with_box_{idx}.png"))

if __name__ == "__main__":
    dataset_path = "C:\\Users\\adhar\\OneDrive\\Desktop\\bounding boxes\\train\\train"
    save_directory = "C:\\Users\\adhar\\OneDrive\\Desktop\\bounding boxes\\saved_images"

    dataset = AppleDataset(root_dir=dataset_path, transforms=None)
    print("Dataset size:", len(dataset))

    num_images_to_process = min(668, len(dataset))
    for i in range(num_images_to_process):
        dataset.save_image_with_boxes(i, save_directory)
        print(f"Processed and saved image {i}")
