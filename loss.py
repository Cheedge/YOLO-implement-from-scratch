import torch
import torch.nn as nn
from IoU import IntersectionOverUnion

class loss(nn.Module):
    """
    calculate YOLOv1 loss
    Input:
    pred: (tensor) (N, S*S*(B*5+C)) N is num_batch. 
    targ: (tensor) (N, S, S, 5+C)

    Parameters:
    S: image grids dimension, here is 7*7
    Boxes: num of predict bounding boxes, here is 2 for every cell
    Classes: num of total classes, here is 20 categries.

    Output:
    loss: (float) total loss by MSELoss
    """

    def __init__(self, S=7, Boxes=2, Classes=20):
        super(loss, self).__init__()
        self.S = S
        self.B = Boxes
        self.C = Classes
        self.mse = nn.MSELoss(reduction='sum')
        self.lambda_coord = 5
        self.lambda_nobody = 0.5

    def forward(self,pred, targ):
        # input pred: (N, S*S*(5*B+C))
        pred = pred.reshape(-1, self.S, self.S, 5*self.B+self.C)
        # 1, class loss
        # (N, S, S, 30)->(N, S, S, 1)
        pred_c = pred[..., 0:19]
        targ_c = targ[..., 0:19]

        obj_cells = targ[..., 24:25] # exist obj 1, no exist 0
        loss_class = self.mse(
                torch.flatten(obj_cells * pred_c, end_dim=2),
                torch.flatten(obj_cells * targ_c, end_dim=2)
                )
        # 2, localization (coordinate) loss
        iou0 = IntersectionOverUnion(pred[..., 20:24], targ[..., 20:24],"centers").unsqueeze(0)
        iou1 = IntersectionOverUnion(pred[..., 25:29], targ[..., 20:24],"centers").unsqueeze(0)
        # compare iou0 and iou1 for every cell
        iou_max, iou_max_cell = torch.max(iou0, iou1, dim=0)

        # pred_b* is (N, S, S, 4)
        pred_b0_coord = (1-iou_max_cell) * pred[..., 20:24]
        pred_b1_coord = iou_max_cell * pred[..., 25:29]
        pred_b0_coord=torch.sign(pred_b0_coord) * torch.sqrt(torch.abs(pred_b0_coord[..., 22:24]))
        pred_b1_coord=torch.sign(pred_b1_coord) * torch.sqrt(torch.abs(pred_b1_coord[..., 27:29]))
        loss_coord = self.mse(
                torch.flatten(pred_b0_coord),
                torch.flatten(targ[..., 20:24])
                )
        loss_coord += self.mse(
                torch.flatten(pred_b1_coord),
                torch.flatten(targ[..., 20:24])
                )

        # 3, confidence loss (exist obj)
        pred_b1_obj = obj_cells * (iou_max_cell * pred[..., 29:30])
        pred_b0_obj = obj_cells * ((1-iou_max_cell) * pred[..., 24:25])
        targ_obj = obj_cells * targ[..., 24:25]

        loss_conf_obj = self.mse(
                torch.flatten(pred_b1_obj, start_dim=1),
                torch.flatten(targ_obj, start_dim=1)
                )
        loss_conf_obj += self.mse(
                torch.flatten(pred_b0_obj, start_dim=1),
                torch.flatten(targ_obj, start_dim=1)
                )
                
        # 4, confidence loss (non-exist obj)
        noobj_cells = 1 - obj_cells
        pred_b1_noobj = noobj_cells * (iou_max_cell * pred[..., 29:30])
        pred_b0_noobj = noobj_cells * ((1-iou_max_cell) * pred[..., 24:25])
        targ_noobj = noobj_cells * targ[..., 24:25]

        loss_conf_noobj = self.mse(
                torch.flatten(pred_b1_noobj, start_dim=1),
                torch.flatten(targ_noobj, start_dim=1)
                )
        loss_conf_noobj += self.mse(
                torch.flatten(pred_b0_noobj, start_dim=1),
                torch.flatten(targ_noobj, start_dim=1)
                )

        # total loss
        loss = loss_class + self.lambda_coord * loss_coord + loss_conf_obj + self.lambda_nobody * loss_conf_noobj

        return loss
