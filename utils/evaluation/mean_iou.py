import numpy as np
from keras.metrics import MeanIoU
from utils.evaluation.helpers import pred
# from tensorflow.keras.metrics import MeanIoU


# def mean_iou_score(model,X_val,y_val,n_classes ):

#     y_acc , y_pred = pred(X_val,y_val,model , len(y_val))
#     IOU_ref = MeanIoU(num_classes=n_classes)
#     IOU_ref.update_state(y_acc , y_pred)


#     values = np.array(IOU_ref.get_weights()).reshape(n_classes,n_classes)


#     class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[1,0]+ values[2,0]+ values[3,0] + values[4,0])
#     class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1]+ values[2,1]+ values[3,1] + values[4,1])
#     class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4]+ values[0,2]+ values[1,2]+ values[3,2]  +values[4,2] )
#     class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[0,3]+ values[1,3]+ values[2,3] + values[4,3])
#     class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2]  +values[4,3] + values[0,4]+ values[1,4]+ values[2,4] + values[3,4])

#     print()
#     print("IoU for class 0 is: ", class1_IoU)
#     print("IoU for class 1 is: ", class2_IoU)
#     print("IoU for class 2 is: ", class3_IoU)
#     print("IoU for class 3 is: ", class4_IoU)
#     print("IoU for class 4 is: ", class5_IoU)
#     print("Mean Iou: ", (class1_IoU + class2_IoU + class3_IoU + class4_IoU + class5_IoU)/4)
#     print()
#     return(class1_IoU, class2_IoU, class3_IoU, class4_IoU, class5_IoU,(class1_IoU + class2_IoU + class3_IoU + class4_IoU + class5_IoU)/4)

# import numpy as np
# from tensorflow.keras.metrics import MeanIoU

def mean_iou_score(model, X_val, y_val, n_classes):
    """
    Compute per-class IoU and mean IoU for a single model.

    Returns:
        dict with keys:
        - class_0, class_1, ..., class_{n_classes-1}
        - mean_iou
    """

    # Predict
    y_true, y_pred = pred(X_val, y_val, model, len(y_val))

    # Compute confusion matrix via MeanIoU
    iou_metric = MeanIoU(num_classes=n_classes)
    iou_metric.update_state(y_true, y_pred)

    cm = np.array(iou_metric.get_weights()).reshape(n_classes, n_classes)

    results = {}
    ious = []

    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0

        # results[f"class_{c}"] = float(iou)
        ious.append(iou)

        print(f"IoU for class {c}: {iou:.6f}")

    mean_iou = float(np.mean(ious))
    results["mean_iou"] = mean_iou

    print(f"Mean IoU: {mean_iou:.6f}\n")


    return {
        "per_class_iou": ious,
        "mean_iou": mean_iou,
        # "confusion_matrix": conf_matrix
    }
    # return results


# def mean_iou_score_ensemble(y_preds,y_actual,n_classes = 5 ):

#     y1 = []

#     for i in range(0,len(y_preds)):

#         y1.append(y_preds[i].reshape(256,256,1))

#     y1 = np.array(y1)

#     IOU_ref = MeanIoU(num_classes=n_classes)
#     IOU_ref.update_state(y_actual , y1)


#     values = np.array(IOU_ref.get_weights()).reshape(n_classes,n_classes)



#     class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[1,0]+ values[2,0]+ values[3,0] + values[4,0])
#     class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1]+ values[2,1]+ values[3,1] + values[4,1])
#     class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4]+ values[0,2]+ values[1,2]+ values[3,2]  +values[4,2] )
#     class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[0,3]+ values[1,3]+ values[2,3] + values[4,3])
#     class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2]  +values[4,3] + values[0,4]+ values[1,4]+ values[2,4] + values[3,4])

#     print()
#     print("IoU for class 0 is: ", class1_IoU)
#     print("IoU for class 1 is: ", class2_IoU)
#     print("IoU for class 2 is: ", class3_IoU)
#     print("IoU for class 3 is: ", class4_IoU)
#     print("IoU for class 4 is: ", class5_IoU)
#     print("Mean Iou: ", (class1_IoU + class2_IoU + class3_IoU + class4_IoU + class5_IoU)/4)
#     print()
#     return(class1_IoU, class2_IoU, class3_IoU, class4_IoU, class5_IoU,(class1_IoU + class2_IoU + class3_IoU + class4_IoU + class5_IoU)/4)


# import numpy as np
# from tensorflow.keras.metrics import MeanIoU

def mean_iou_score_ensemble(y_pred, y_true, n_classes=5, verbose=True):
    """
    y_pred: (N, H, W) or (N, H, W, 1)
    y_true: (N, H, W, 1)
    """

    if y_pred.ndim == 4:
        y_pred = y_pred.squeeze(-1)

    if y_true.ndim == 4:
        y_true = y_true.squeeze(-1)

    metric = MeanIoU(num_classes=n_classes)
    metric.update_state(y_true, y_pred)

    conf_matrix = metric.get_weights()[0].reshape(n_classes, n_classes)

    ious = {}
    for c in range(n_classes):
        TP = conf_matrix[c, c]
        FP = conf_matrix[:, c].sum() - TP
        FN = conf_matrix[c, :].sum() - TP
        denom = TP + FP + FN
        ious[c] = TP / denom if denom > 0 else 0.0

    mean_iou = np.mean(list(ious.values()))

    if verbose:
        print()
        for c, v in ious.items():
            print(f"IoU for class {c}: {v:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        print()

    return {
        "per_class_iou": ious,
        "mean_iou": mean_iou,
        # "confusion_matrix": conf_matrix
    }
