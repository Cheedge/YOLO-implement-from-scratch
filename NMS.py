import torch
from IoU import IntersectionOverUnion

def Non_Max_Suppression(bbox_prds, IoU_threshold, box_format="corners"):
    """
    keep not too much overlaped box

    Input:
    bbox_prds: (lsit) for all class all prob boxes, formate as:
                [[class, class_prob, xyHW], [class, class_prob, xyHW], ...]
    IoU_threshold: (float) decide wether tow boxes are too much overlaped
    box_format: (str) same as IntersectionOverUnion
    Output:
    NMS_bbox: (list) return all the not too much overlaped boxes.
    """

    # bbox_prds is a list [bouding boxs]= [[class, class_prob, xyHW], [class, class_prob, xyHW], ...]
    assert type(bbox_prds) == list
    bbox_highprob = [box for box in bbox_prds if box[1] > class_prob_thd] 
    bbox_highprob = sorted(bbox_prds, key= lambda x: x[1], reverse=True)

    NMS_bbox = []
    while bbox_highprob:
        choose_box = bbox_highprob.pop(0)

        # keep diff class case and iou < threshold case in bbox_highprob
        bbox_highprob = [ box for box in bbox_highprob
                            if box[0] != choose_box[0] or
                            IntersectionOverUnion(box[2:], choose_box[2:]) 
                            < IoU_threshold ]
        NMS_bbox.append(choose_box)

        return NMS_bbox
