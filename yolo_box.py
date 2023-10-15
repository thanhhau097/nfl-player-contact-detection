import pandas as pd

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

data = pd.read_csv("data/train_baseline_helmets_kfold.csv")
process_id = 7
view = 'Sideline' # 'Endzone' 

data = data[len(data)*process_id//8:len(data)*(process_id+1)//8]
data = data[data.view == view]

import os
import glob

paths = {}
for filename in glob.glob("data/frames/*/*.jpg"):
    paths[os.path.basename(filename)] = filename

from tqdm import tqdm

df_paths = []
for index, row in tqdm(data.iterrows()):
    path = row["video"] + "_" + "0" * (4 - len(str(row["frame"]))) + str(row["frame"])
    df_paths.append(path)

data["path"] = [path + ".jpg" for path in df_paths]

intersection_paths = list(set(data["path"]).intersection(set(paths.keys())))

from PIL import Image
import numpy as np

import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # or yolov5n - yolov5x6, custom
# model = YOLO("yolov5l.pt")
model.conf = 0.1  # NMS confidence threshold
model.iou = 0.70  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = [0]  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2

data["x1"] = 0
data["y1"] = 0
data["x2"] = 0
data["y2"] = 0


from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, path_dict, path_list):
        self.path_dict = path_dict
        self.path_list = path_list
        
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, i):
        image = Image.open(self.path_dict[self.path_list[i]])
        image = np.array(image)
        return image, self.path_list[i] # .transpose(2, 0, 1)

ds = CustomDataset(paths, intersection_paths)

dl = DataLoader(ds, batch_size=320, shuffle=False, collate_fn=lambda x: x)

for item in tqdm(dl):
    res = model([i[0] for i in item])
    
    # for idx, r in enumerate(res):
    for idx, xywh in enumerate(res.xywh):
        # xywh = r.boxes.xywh
        for i, row in data[data["path"] == item[idx][1]].iterrows():
            x1, y1, x2, y2 = int(row["left"]), int(row["top"]), int(row["left"]) + int(row["width"]), int(row["top"]) + int(row["height"])

            best_iou, best_box = 0, []
            for box in xywh.int().cpu().numpy():
                x, y, w, h = box[:4]
                box_x1 = x - w // 2
                box_y1 = y - h // 2
                box_x2 = x + w // 2
                box_y2 = y + h // 2
                try:
                    iou = get_iou(
                        {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        {"x1": box_x1, "y1": box_y1, "x2": box_x2, "y2": y}
                    )
                except:
                    iou = -1
                    print("Error", [i[1] for i in item])

                if iou > best_iou:
                    best_iou = iou
                    best_box = [box_x1, box_y1, box_x2, box_y2]

            if len(best_box) == 0:
                best_box = [x1 - 128, y1 - 128, x1 + 128, y1 + 128]

            data.loc[i, "x1"] = best_box[0]
            data.loc[i, "y1"] = best_box[1]
            data.loc[i, "x2"] = best_box[2]
            data.loc[i, "y2"] = best_box[3]

data.to_csv(f"data/train_baseline_helmets_kfold_yolox_boxes_{process_id}_{view}.csv")

