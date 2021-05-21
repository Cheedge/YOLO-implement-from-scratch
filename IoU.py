import torch

def IntersectionOverUnion(boxes_prds, boxes_trgt, box_format="centers"):
    """
    Calculate 'Intersection of Union'. which used
    for train predict bounding box more closer to the target bounding
    box.
    Parameters: boxes_prds(Tensor):Prediction of Bounding box. (:, BITCH_SIZE, 4)
                boxes_trgt(Tensor):Targets of Bounding box. (:, BITCH_SIZE, 4)
                box_format(str): centers/corners. which use (x, y, H, W) or (x1, y1, x2, y2)
    Return:
        IoU scores for all examples(Tensor).
    """
    # boxes_prds: (N, 4) where N is numof Bounding boxes.
    # boxes_labels: (N, 4)
    # use [..., 0:1] same as [:, 0:1], but : can take as one dim,
    # ... means all dims above
    # and [..., 0:1] means colum list, [..., 0] means row list.
    if box_format == "centers":
        box1_xc = boxes_prds[..., 0:1]
        box1_yc = boxes_prds[..., 1:2]
        box1_H = boxes_prds[..., 2:3]
        box1_W = boxes_prds[..., 3:4]
        box2_xc = boxes_trgt[..., 0:1]
        box2_yc = boxes_trgt[..., 1:2]
        box2_H = boxes_trgt[..., 2:3]
        box2_W = boxes_trgt[..., 3:4]

        box1_x1 = box1_xc - box1_W / 2.0
        box1_y1 = box1_yc - box1_H / 2.0
        box1_x2 = box1_xc + box1_W / 2.0
        box1_y2 = box1_yc + box1_H / 2.0
        box2_x1 = box2_xc - box2_W / 2.0
        box2_y1 = box2_yc - box2_H / 2.0
        box2_x2 = box2_xc + box2_W / 2.0
        box2_y2 = box2_yc + box2_H / 2.0

    if box_format == "corners":
        box1_x1 = boxes_prds[..., 0:1]
        box1_y1 = boxes_prds[..., 1:2]
        box1_x2 = boxes_prds[..., 2:3]
        box1_y2 = boxes_prds[..., 3:4]
        box2_x1 = boxes_trgt[..., 0:1]
        box2_y1 = boxes_trgt[..., 1:2]
        box2_x2 = boxes_trgt[..., 2:3]
        box2_y2 = boxes_trgt[..., 3:4]

    I_x1 = torch.max(box1_x1, box2_x1)
    I_y1 = torch.max(box1_y1, box2_y1)
    I_x2 = torch.min(box1_x2, box2_x2)
    I_y2 = torch.min(box1_y2, box2_y2)

    # clamp(0) makes less than 0 value to 0, so non-intersection will be 0
    Intersection = (I_x2 - I_x1).clamp(0) * (I_y2 - I_y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    epsilon = 1e-6

    return Intersection / (box1_area + box2_area - Intersection + epsilon)
