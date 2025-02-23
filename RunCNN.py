from Visualize import *
from torch.utils.data import DataLoader, TensorDataset
from Parameters import *
from FacialDetector import *
from Visualize import *
import torch
import os
import numpy as np



params: Parameters = Parameters()
params.dim_window = 64
params.overlap = 0.3
params.number_positive_examples = 5813  # numarul exemplelor pozitive
params.number_negative_examples = 11599  # numarul exemplelor negative
params.image_scale = [1.0,  0.96, 0.9, 0.86, 0.84, 0.8,
                    0.76, 0.7, 0.68, 0.66, 0.64, 0.62, 0.6,
                     0.56, 0.5, 0.48, 0.46, 0.44, 0.42, 0.4,
                      0.38, 0.36, 0.34, 0.32, 0.3, 0.25, 0.2]
params.threshold = 5.5 # toate ferestrele cu scorul > threshold si maxime locale devin detect
params.epochs = 20
# ii
params.has_annotations = False
params.use_flip_images = False  # adauga imaginile cu fete oglindite


if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)

positive_folder = params.dir_pos_examples  # ADD PATH
negative_folder = params.dir_neg_examples  # ADD PATH

def load_images_from_folder(folder, label, image_size=(params.dim_window, params.dim_window)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if filename.endswith('.jpg'):
            img = cv.imread(img_path)
            img = cv.resize(img, image_size)
            images.append(img)
            labels.append(label)
    return images, labels


positive_folder = params.dir_pos_examples  # ADD PATH
negative_folder = params.dir_neg_examples  # ADD PATH
print("Am incarcat exemplele pozitive")
positive_images, positive_labels = load_images_from_folder(positive_folder, 1)
print("Am incarcat exemplele negative")
negative_images, negative_labels = load_images_from_folder(negative_folder, 0)

# concatenam  imaginile pozitive si  cele negative
images = np.array(positive_images + negative_images, dtype=np.float32)
labels = np.array(positive_labels + negative_labels)
images_rgb = [cv.cvtColor(img, cv.COLOR_GRAY2RGB) if len(img.shape) == 2 else img for img in images]

# normalizare + array
images_np = np.array(images_rgb, dtype=np.float32) / 255.0  # Normalizeaza la [0, 1]
# conversie la tensori
images_tensor = torch.tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2) #[batch_size, channels, height, width]
labels_tensor = torch.tensor(labels, dtype=torch.long)
# legam imaginile de etichetele lor
dataset = TensorDataset(images_tensor, labels_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
# Antrenare
facial_detector.train_classifier(train_loader)


detections, scores, file_names = facial_detector.run()

ans_output_dir = "../../331_Andrei_Popa_rezultate/task1"
os.makedirs(ans_output_dir, exist_ok=True)

np.save("../../331_Andrei_Popa_rezultate/task1/detections_all_faces.npy", detections)
np.save("../../331_Andrei_Popa_rezultate/task1/scores_all_faces.npy", scores)
np.save("../../331_Andrei_Popa_rezultate/task1/file_names_all_faces.npy", file_names)


# TASK2 - consideram ca Dexter este peste tot
ans_output_dir = "../../331_Andrei_Popa_rezultate/task2"
os.makedirs(ans_output_dir, exist_ok=True)

np.save("../../331_Andrei_Popa_rezultate/task2/detections_dexter.npy", detections)
np.save("../../331_Andrei_Popa_rezultate/task2/scores_dexter.npy", scores)
np.save("../../331_Andrei_Popa_rezultate/task2/file_names_dexter.npy", file_names)


if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)

else:
    show_detections_without_ground_truth(detections, scores, file_names, params)
