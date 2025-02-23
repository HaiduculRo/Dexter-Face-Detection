import timeit
import numpy as np
from ultralytics import YOLO
import ntpath
import os
from FacialDetector import *
import Parameters
from Visualize import *

def custom_predict(model_path, test_images_path):
    # Încarcă modelul YOLO antrenat
    model = YOLO(model_path)
    test_files = [f"{test_images_path}/{file}" for file in os.listdir(test_images_path) if file.endswith(".jpg")]

    detections = None
    scores = np.array([])
    file_names = np.array([])

    num_test_images = len(test_files)
    for i, image_path in enumerate(test_files):
        start_time = timeit.default_timer()

        # Predicția pe imagine
        results = model.predict(source=image_path, save=False)

        image_detections = []
        image_scores = []

        for box in results[0].boxes:
            x_center, y_center, width, height = box.xywh[0].tolist()
            xmin = int(x_center - width / 2)
            ymin = int(y_center - height / 2)
            xmax = int(x_center + width / 2)
            ymax = int(y_center + height / 2)

            image_detections.append([xmin, ymin, xmax, ymax])
            image_scores.append(box.conf.item())

        if len(image_scores) > 0:
            if detections is None:
                detections = np.array(image_detections)
            else:
                detections = np.concatenate((detections, np.array(image_detections)))

            scores = np.append(scores, image_scores)
            short_name = ntpath.basename(image_path)
            image_names = [short_name for _ in range(len(image_scores))]
            file_names = np.append(file_names, image_names)

        end_time = timeit.default_timer()
        print(f"Timpul de procesare al imaginii de testare {i + 1}/{num_test_images} este {end_time - start_time:.3f} sec.")

    return np.array(detections), np.array(scores), np.array(file_names)

params = Parameters()

model_path = "runs/detect/train11/weights/best.pt"
test_images_path = "../../testare_t2"   #schimbat path

# Predictii
detections, scores, file_names = custom_predict(model_path, test_images_path)

ans_output_dir = "../../331_Andrei_Popa_rezultate/task1_YOLO"
os.makedirs(ans_output_dir, exist_ok=True)
np.save(os.path.join(ans_output_dir, "detections_all_faces.npy"), detections)
np.save(os.path.join(ans_output_dir, "scores_all_faces.npy"), scores)
np.save(os.path.join(ans_output_dir, "file_names_all_faces.npy"), file_names)

