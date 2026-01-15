
from utils.dataprocessing.create_image import process_x, process_x2
from utils.dataprocessing.create_mask import process_y, process_y2
from utils.dataprocessing.further_processing import further_process
import segmentation_models_3D as sm
from utils.unet.build_unet import build_unet
from utils.evaluation.helpers import show_predictions,display,create_mask
from utils.evaluation.mean_iou import mean_iou_score, mean_iou_score_ensemble
import numpy as np
import os

import cv2

dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.20, 0.20, 0.20, 0.20,0.20]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = ["accuracy",sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
lr =1e-3


path_img_agu_test ="./diadem_data/preprocessed/test/"
path_label_agu_test = "./diadem_data/preprocessed/truth/"

save_dir = "./results_images/"

img_test = []
for f in sorted(os.listdir(path_img_agu_test)):
  img_test.append(os.path.join(path_img_agu_test ,f))


labels_test = []
for f in sorted(os.listdir(path_label_agu_test)):
  labels_test.append(os.path.join(path_label_agu_test,f))


print(f"nb image : {len(img_test)} ")
print(f"nb mask : {len(labels_test)} ")

X_test_image_normal = process_x(img_test)
y_test_image_normal = process_y(labels_test)

X_test_image_HSV = process_x(img_test , con = "HSV" )
y_test_image_HSV = process_y(labels_test)

X_test_image_YUV = process_x(img_test , con = "YUV")
y_test_image_YUV = process_y(labels_test)



path_model ="models_training_all_dataset/"
# prefix = "unet_"
prefix="only_diadem_unet_"
prefix ="diad_val_truthunet_"
train_normal = "normal"
train_HSV = "HSV"
train_YUV = "YUV"


model_arc_normal = build_unet((256,256, 3),loss = total_loss , lr = lr , metrics = metrics )
model_arc_normal.load_weights(path_model+prefix+train_normal+".weights.h5")
model_arc_normal.eval = False  # just clarity
l_score_normal =  mean_iou_score(model_arc_normal,X_test_image_normal,y_test_image_normal,5)

model_arc_HSV = build_unet((256,256, 3),loss = total_loss , lr = lr , metrics = metrics )
model_arc_HSV.load_weights(path_model+prefix+train_HSV+".weights.h5")
l_score_HSV =  mean_iou_score(model_arc_HSV,X_test_image_HSV,y_test_image_HSV,5)

model_arc_YUV = build_unet((256,256, 3),loss = total_loss , lr = lr , metrics = metrics )
model_arc_HSV.load_weights(path_model+prefix+train_YUV+".weights.h5")
l_score_YUV = mean_iou_score(model_arc_YUV,X_test_image_YUV,y_test_image_YUV,5)

preds_normal = model_arc_normal.predict(X_test_image_normal)



os.makedirs(save_dir, exist_ok=True)

for i, img_path in enumerate(img_test):
  fname = os.path.basename(img_path)
  out_name = fname.replace(".jpg", ".png")

  orig_img = cv2.imread(img_path)
  H, W = orig_img.shape[:2]


  # Use your proven logic
  pred_mask = create_mask(preds_normal[i:i+1])  # (256,256,1)
  pred_mask_np = pred_mask.numpy().squeeze().astype(np.uint8)  # (256,256)

  pred_resized = cv2.resize(
    pred_mask_np,
    (W, H),
    interpolation=cv2.INTER_NEAREST
  )


  cv2.imwrite(
    os.path.join(save_dir, out_name),
    pred_resized
  )
  # cv2.imwrite(
  #     os.path.join(save_dir, out_name),
  #     pred_mask_np
  # )

  print(f"Saved prediction: {out_name}, shape={pred_mask_np.shape}")




# os.makedirs(save_dir, exist_ok=True)

# preds_normal = model_arc_normal.predict(X_test_image_normal)
# pred_labels = np.argmax(preds_normal, axis=-1)  # (N, 256, 256)

# for i, img_path in enumerate(img_test):
#     fname = os.path.basename(img_path)
#     out_name = fname.replace(".jpg", ".png")

#     cv2.imwrite(
#         os.path.join(save_dir, out_name),
#         pred_labels[i].astype(np.uint8)
#     )

#     print(f"Saved prediction: {out_name}")


# pred2 = model_arc_HSV.predict(X_test_image_HSV)
# pred3 = model_arc_YUV.predict(X_test_image_YUV)


# print(pred1)
# import os
# import numpy as np

# save_dir = "predictions/normal/"
# os.makedirs(save_dir, exist_ok=True)
# pred1_labels = np.argmax(pred1, axis=-1)

# print(pred1_labels)


# for i in range(pred1_labels.shape[0]):
#     mask = pred1_labels[i].astype(np.uint8)

#     cv2.imwrite(
#         os.path.join(save_dir, f"pred_{i:03d}.png"),
#         mask
#     )


# """
# Ensemble of the three outputs

# """

# preds=np.array([pred1, pred2, pred3])

# weights = [0.4, 0.3, 0.3]

# w = [1,1,1]
# weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
# weighted_ensemble_prediction = np.argmax(weighted_preds, axis=3)

# #(1, 256, 256, 3) (256, 256, 1) (256, 256, 1)

# print()
# print("________________________________________________________________")
# print()
# print("weighted ensemble " )
# print()
# print("________________________________________________________________")
# print()


# for i in range(0,5):

#     display([X_test_image_normal[i:i+1] ,y_test_image_normal[i] ,  weighted_ensemble_prediction[i].reshape(256,256,1)])

# l_score_ensemble_weigthed = mean_iou_score_ensemble(weighted_ensemble_prediction,y_test_image_normal,n_classes = 5 )


# un_weighted_preds = np.tensordot(preds, w, axes=((0),(0)))
# un_weighted_ensemble_prediction = np.argmax(un_weighted_preds, axis=3)

# print()
# print("________________________________________________________________")
# print()
# print("un - weighted ensemble " )
# print()
# print("________________________________________________________________")
# print()
# for i in range(0,5):

#     display([X_test_image_normal[i:i+1] ,y_test_image_normal[i] ,  un_weighted_ensemble_prediction[i].reshape(256,256,1)])

# l_score_ensemble_unweigthed = mean_iou_score_ensemble(un_weighted_ensemble_prediction ,y_test_image_normal,n_classes = 5 )


