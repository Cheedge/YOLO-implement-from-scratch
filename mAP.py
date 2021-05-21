import torch
from collections import Counter
from IoU import IntersectionOverUnion

def meanAveragePrecision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20):
    """
    AP@n = sum_i=1^N (P@i * (R@(i+1) - R@i)) = int P@i * S
    Input:
    pred_boxes: (list) predicted bounding boxes of all image all classes. formate as  
                [[train_idx, class_pred, prob_score, xyHW], [], []]
    true_boxes: (list) ture boxes of all image all classes. format as above
    iou_threshold: (float) if IoU(pred, targ)> iou_threshold, it is the True Positive(TP).
    box_format: (str) corners or centers same as IntersectionOverUnion function
    num_classes: (int) all classes in all images.
    Output:
    mean Average Precision: (float)
    """
    # pred_boxes (list): [[train_idx, class_pred, prob_score, xyHW], [], []]
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detect_boxes = []
        exist_boxes = []
        
        # assign diff box to diff class.
        for det_box in pred_boxes:
            if det_box[1] == c:
                detect_boxes.append(det_box)

        # assign diff tgt box to diff class.
        for tgt_box in true_boxes:
            if tgt_box[1] == c:
                exist_boxes.append(tgt_box)


        # Counter return a dict, {key: num_occur}
        # img0 has 3 bbox
        # img1 has 5 bbox
        # amount_bboxes={0:3, 1:5}
        unmatched_trueboxes = Counter([tb[0] for tb in exist_boxes])
        
        for key, val in unmatched_trueboxes.items():
            unmatched_trueboxes[key] = torch.zeros(val)

        detect_boxes.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detect_boxes))
        FP = torch.zeros(len(detect_boxes))
        num_true_bboxes = len(exist_boxes)

        # here check if the class is exist, if not continue to next loop
        if num_true_bboxes ==0:
            continue

        # loop pred_box
        for det_idx, detect in enumerate(detect_boxes):
            iou_max = 0
            # in same image
            #if detect[0] == exist_boxes[0]:
            ground_true_boxes = [box for box in exist_boxes if detect[0] == box[0]]
            for truebox_idx, truebox in enumerate(ground_true_boxes):
                if unmatched_trueboxes[detect[0]][truebox_idx] == 0:
                    iou = IntersectionOverUnion(torch.tensor(detect[3:]),
                            torch.tensor(truebox[3:]), box_format=box_format)

                    if iou > iou_max:
                        iou_max = iou
                        truebox_max_idx = truebox_idx

            if iou_max > iou_threshold:
                TP[det_idx] = 1
                # detect[0] is the image idx
                unmatched_trueboxes[detect[0]][truebox_max_idx]=1
            else:
                FP[det_idx] = 1
            # cumsum TP, FP
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            precision = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            recall = TP_cumsum / (num_true_bboxes + epsilon)
            precision = torch.cat((torch.tensor([1]), precision))
            recall = torch.cat((torch.tensor([0]), recall))

            average_precisions.append(torch.trapz(precision, recall))
    return sum(average_precisions) / len(average_precisions)
