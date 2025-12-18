# from utils.cross_val_results.k_fold import k_fold
import argparse
import os


import numpy as np
from utils.dataprocessing.create_image import process_x, process_x2
from utils.dataprocessing.create_mask import process_y, process_y2
from utils.dataprocessing.further_processing import further_process
import segmentation_models_3D as sm
from utils.unet.build_unet import build_unet
from utils.evaluation.helpers import show_predictions,display
from utils.evaluation.mean_iou import mean_iou_score, mean_iou_score_ensemble


import json

path_model ="models_training_all_dataset/"
file = "results.txt"



parser = argparse.ArgumentParser()

parser.add_argument('--images_path', type=str, default = './',
                    help='Path where the images folder is stored')

parser.add_argument('--masks_path', type=str, default = './',
                    help='Path where the masks folder is stored')

parser.add_argument('--epochs', type=int, default = 250,
                    help='Number of Epochs for training')

parser.add_argument('--batch', type=int, default = 4,
                    help='Batch Size for Mini Batch Training')

# parser.add_argument('--n_splits', type=int, default = 6,
#                     help='Number of folds for training')

parser.add_argument('--lr', type=float, default = 1e-3,
                    help='Learning rate for training')

parser.add_argument('--show', type=bool, default = False,
                    help='Showing the comparison among original, ground-truth and predicted images')



parser.add_argument('--test_img_path', type=str, default = './',
                    help='Path where the masks folder is stored')


parser.add_argument('--test_masks_path', type=str, default = './',
                    help='Path where test masks folder is stored')

args = parser.parse_args()



### load training imgs
path_img_agu =  args.images_path
img = []

for f in sorted(os.listdir(path_img_agu)):
  img.append(os.path.join(path_img_agu ,f))



path_label_agu =  args.masks_path
labels = []

for f in sorted(os.listdir(path_label_agu)):
  labels.append(os.path.join(path_label_agu,f))

print(f"nb image : {len(img)} ")
print(f"nb mask : {len(labels)} ")


###############   load test images
path_img_agu_test =  args.test_img_path

img_test = []
for f in sorted(os.listdir(path_img_agu_test)):
  img_test.append(os.path.join(path_img_agu_test ,f))

path_label_agu_test =  args.test_masks_path

labels_test = []
for f in sorted(os.listdir(path_label_agu_test)):
  labels_test.append(os.path.join(path_label_agu_test,f))


print(f"nb image : {len(img_test)} ")
print(f"nb mask : {len(labels_test)} ")




def log_results_json(path, fold, scores, model_name):
    entry = {
        "fold": fold,
        "model": model_name,
        "per_class_iou": scores["per_class_iou"],
        "mean_iou": scores["mean_iou"]
    }

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)




X = np.array(img)
y = np.array(labels)

np.random.seed(42)
indices = np.random.permutation(len(X))

val_size = int(0.1 * len(X))
val_idx = indices[:val_size]
train_idx = indices[val_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]


X_train_image_normal = process_x2(X)
y_train_image_normal  = process_y2(y, cat = True )

X_train_image_HSV = process_x2(X , con = "HSV" )
y_train_image_HSV  = process_y2(y, cat = True )

X_train_image_YUV = process_x2(X, con = "YUV")
y_train_image_YUV  = process_y2(y, cat = True )



#val
X_val_image_normal = process_x2(X_val)
y_val_image_normal  = process_y2(y_val, cat = True )

X_val_image_HSV = process_x2(X_val , con = "HSV" )
y_val_image_HSV  = process_y2(y_val, cat = True )

X_val_image_YUV = process_x2(X_val, con = "YUV")
y_val_image_YUV  = process_y2(y_val, cat = True )


#test

X_test_image_normal = process_x(img_test)
y_test_image_normal = process_y(labels_test)

X_test_image_HSV = process_x(img_test , con = "HSV" )
y_test_image_HSV = process_y(labels_test)

X_test_image_YUV = process_x(img_test , con = "YUV")
y_test_image_YUV = process_y(labels_test)




STEPS_PER_EPOCH = len(X) // args.batch
VALIDATION_STEPS = len(X_val) // args.batch






dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.20, 0.20, 0.20, 0.20,0.20]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
metrics = ["accuracy",sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]



train_normal , val_normal = further_process(X_train_image_normal,y_train_image_normal,X_val_image_normal,y_val_image_normal)

train_HSV , val_HSV = further_process(X_train_image_HSV,y_train_image_HSV,X_val_image_HSV,y_val_image_HSV)

train_YUV , val_YUV = further_process(X_train_image_YUV,y_train_image_YUV,X_val_image_YUV,y_val_image_YUV)


epochs =  args.epochs
lr = args.lr

model_arc_normal = build_unet((256,256, 3),loss = total_loss , lr = lr , metrics = metrics )
model_arc_normal.fit(train_normal,
            validation_data=val_normal,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            epochs=epochs,
            )

l_score_normal =  mean_iou_score(model_arc_normal,X_test_image_normal,y_test_image_normal,5)


model_arc_normal.save(path_model+"unet_normal.h5")
model_arc_normal.save_weights(f"models/unet_normal.weights.h5")



### HSV model 
model_arc_HSV = build_unet((256,256, 3),loss = total_loss , lr = lr , metrics = metrics )

model_arc_HSV.fit(train_HSV,
            validation_data=val_HSV,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            epochs=epochs,
            )

model_arc_HSV.save(path_model+"unet_hsv.h5")
model_arc_HSV.save_weights(f"models/unet_HSV.weights.h5")

l_score_HSV =  mean_iou_score(model_arc_HSV,X_test_image_HSV,y_test_image_HSV,5)

#### YUV model 


model_arc_YUV = build_unet((256,256, 3),loss = total_loss , lr = lr , metrics = metrics )

model_arc_YUV.fit(train_YUV,
            validation_data=val_YUV,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            epochs=epochs,
            )


model_arc_YUV.save(path_model+"unet_yuv.h5")

model_arc_YUV.save_weights(f"models/unet_YUV.weights.h5")

l_score_YUV = mean_iou_score(model_arc_YUV,X_test_image_YUV,y_test_image_YUV,5)




# print(l_score_normal)
# log_results_json(path_model+file,fold,l_score_normal,"normal")
# log_results_json(path_model+file,fold,l_score_HSV,"HSV")
# log_results_json(path_model+file,fold,l_score_YUV,"YUV")
# log_results_json(path_model+file,fold,l_score_ensemble_weigthed,"ensemble_weigthed")
# log_results_json(path_model+file,fold,l_score_ensemble_unweigthed,"ensemble_unweigthed")



# l_score_normal =  mean_iou_score(model_arc_normal,X_test_image_normal,y_test_image_normal,5)



pred1 = model_arc_normal.predict(X_test_image_normal)
pred2 = model_arc_HSV.predict(X_test_image_HSV)
pred3 = model_arc_YUV.predict(X_test_image_YUV)


"""
Ensemble of the three outputs

"""

preds=np.array([pred1, pred2, pred3])

weights = [0.4, 0.3, 0.3]

w = [1,1,1]
weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=3)

#(1, 256, 256, 3) (256, 256, 1) (256, 256, 1)

print()
print("________________________________________________________________")
print()
print("weighted ensemble " )
print()
print("________________________________________________________________")
print()


for i in range(0,5):

    display([X_test_image_normal[i:i+1] ,y_test_image_normal[i] ,  weighted_ensemble_prediction[i].reshape(256,256,1)])

l_score_ensemble_weigthed = mean_iou_score_ensemble(weighted_ensemble_prediction,y_test_image_normal,n_classes = 5 )


un_weighted_preds = np.tensordot(preds, w, axes=((0),(0)))
un_weighted_ensemble_prediction = np.argmax(un_weighted_preds, axis=3)

print()
print("________________________________________________________________")
print()
print("un - weighted ensemble " )
print()
print("________________________________________________________________")
print()
for i in range(0,5):

    display([X_test_image_normal[i:i+1] ,y_test_image_normal[i] ,  un_weighted_ensemble_prediction[i].reshape(256,256,1)])

l_score_ensemble_unweigthed = mean_iou_score_ensemble(un_weighted_ensemble_prediction ,y_test_image_normal,n_classes = 5 )






# path_model ="models/"
# model_arc_normal.save(path_model+"unet_normal_fold{}.h5".format(fold))
# model_arc_HSV.save(path_model+"unet_hsv_fold{}.h5".format(fold))
# model_arc_YUV.save(path_model+"unet_yuv_fold{}.h5".format(fold))

# model_arc_normal.save_weights(f"models/unet_fold_{fold}.weights.h5")
# model_arc_HSV.save_weights(f"models/unet_fold_{fold}.weights.h5")
# model_arc_YUV.save_weights(f"models/unet_fold_{fold}.weights.h5")

# file = "results.txt"

fold = 0 
# print(l_score_normal)
log_results_json(path_model+file,fold,l_score_normal,"normal")
log_results_json(path_model+file,fold,l_score_HSV,"HSV")
log_results_json(path_model+file,fold,l_score_YUV,"YUV")
log_results_json(path_model+file,fold,l_score_ensemble_weigthed,"ensemble_weigthed")
log_results_json(path_model+file,fold,l_score_ensemble_unweigthed,"ensemble_unweigthed")
