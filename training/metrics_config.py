import numpy as np
import torch
from skimage import measure

class mIoUmeter:
    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

        area_tp, area_fp, area_fn = batch_tp_fp_fn(preds, labels, 1)
        self.total_tp += area_tp
        self.total_fp += area_fp
        self.total_fn += area_fn

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        prec = 1.0 * self.total_tp / (np.spacing(1) + self.total_tp + self.total_fp)
        recall = 1.0 * self.total_tp / (np.spacing(1) + self.total_tp + self.total_fn)
        fscore = 2.0 * prec * recall / (np.spacing(1) + prec + recall)
        return float(pixAcc), mIoU, fscore

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0


def batch_tp_fp_fn(predict, target, nclass):
    mini = 1
    maxi = nclass
    nbins = nclass

    # predict = (output.detach().numpy() > 0).astype('int64')  # P
    # target = target.numpy().astype('int64')  # T
    intersection = predict * (predict == target)  # TP

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))

    # areas of TN FP FN
    area_tp = area_inter[0]
    area_fp = area_pred[0] - area_inter[0]
    area_fn = area_lab[0] - area_inter[0]

    # area_union = area_pred + area_lab - area_inter
    assert area_tp <= (area_tp + area_fn + area_fp)
    return area_tp, area_fp, area_fn


class PD_FAmeter:
    def __init__(self):
        super().__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.FA = 0
        self.target = 0

    def update(self, preds_batch, labels_batch, size):
        for _, (preds, labels) in enumerate(zip(preds_batch, labels_batch)):
            predits = np.array((torch.squeeze(preds)).cpu()).astype("int64")
            labelss = np.array((labels).cpu()).astype("int64")

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss, connectivity=2)
            coord_label = measure.regionprops(label)

            self.target += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match = []
            self.dismatch = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            # true_img = np.zeros(predits.shape)
            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)
                        # true_img[coord_image[m].coords[:, 0], coord_image[m].coords[:, 1]] = 1
                        del coord_image[m]
                        break

            # self.dismatch_pixel += (predits - true_img).sum()
            self.dismatch = [
                x for x in self.image_area_total if x not in self.image_area_match
            ]
            self.all_pixel += size[0] * size[1]
            self.FA += np.sum(self.dismatch)
            self.PD += len(self.distance_match)

    def get(self):
        # Final_FA = self.dismatch_pixel / self.all_pixel
        Final_FA = self.FA / self.all_pixel
        Final_PD = self.PD / self.target
        return Final_PD, float(Final_FA)

    def reset(self):
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.FA = 0
        self.target = 0


def batch_pix_accuracy(output, target):
    assert output.shape == target.shape
    output = output.detach().numpy()
    target = target.detach().numpy()

    predict = (output > 0).astype("int64")  # P
    pixel_labeled = np.sum(target > 0)  # T
    pixel_correct = np.sum((predict == target) * (target > 0))  # TP
    assert pixel_correct <= pixel_labeled
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1  # nclass
    nbins = 1  # nclass
    predict = (output.detach().numpy() > 0).astype("int64")  # P
    target = target.numpy().astype("int64")  # T
    intersection = predict * (predict == target)  # TP

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all()
    return area_inter, area_union


class metricWrapper:
    def __init__(self, n_class=1):
        self.miou_meter = mIoUmeter()
        self.pdfa_meter = PD_FAmeter()

    def __call__(self, pred_logits, targets):
        B, C, H, W = targets.shape
        pred_logits = pred_logits
        pred_masks = pred_logits > 0
        self.miou_meter.update(
            pred_masks.reshape(B, 1, H, W).cpu().float(),
            targets.reshape(B, 1, H, W).cpu().float(),
        )
        self.pdfa_meter.update(
            pred_masks.reshape(B, H, W).cpu().float(),
            targets.reshape(B, H, W).cpu().float(),
            [H, W],
        )

    def __str__(self):
        pixAcc, mIoU, fscore = self.miou_meter.get()
        PD, FA = self.pdfa_meter.get()
        info_str = f"mIoU={mIoU*100:.4f}, Fscore={fscore*100:.4f}, PD={PD*100:.4f}, FA={FA*1e6:.4f}"
        return info_str

    def reset(self):
        self.miou_meter.reset()
        self.pdfa_meter.reset()
