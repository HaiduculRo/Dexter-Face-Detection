
from Parameters import *
import numpy as np
import matplotlib.pyplot as plt
import timeit
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import glob
import cv2 as cv
import ntpath

class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=2, dim_window=64):
        super(CNN, self).__init__()
        self.dim_window = dim_window
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        flattened_dim = 64 * (self.dim_window // 8) * (self.dim_window // 8)
        self.fc1 = nn.Linear(flattened_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, cnn_input):
        cnn_input = self.relu(self.conv1(cnn_input))
        cnn_input = self.pool(cnn_input)
        cnn_input = self.relu(self.conv2(cnn_input))
        cnn_input = self.pool(cnn_input)
        cnn_input = self.relu(self.conv3(cnn_input))
        cnn_input = self.pool(cnn_input)

        # Flatten pentru stratul fc1
        cnn_input = cnn_input.reshape(cnn_input.size(0), -1)  # Flatten
        cnn_input = self.relu(self.fc1(cnn_input))
        cnn_input = self.fc2(cnn_input)
        return cnn_input


class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None

    def train_classifier(self, train_loader):
        epochs = self.params.epochs
        max_acc = 0.0

        # Incarca CNN
        cnn_save_file = os.path.join(self.params.dir_save_files,
                                     f'top_cnn_{self.params.dim_window}_neg_{self.params.number_negative_examples}_poz_{self.params.number_positive_examples}_epc_{epochs}.pth')

        if os.path.exists(cnn_save_file):
            self.best_model = torch.load(cnn_save_file)
            print("Exista acest model si urmeaza sa fie incarcat")
            return

        # Parametrii antrenare
        # input_channels = 3 =>> RGB  num_classes=2 =>> Fata/Non-Fata

        model = CNN(input_channels=3, num_classes=2, dim_window=self.params.dim_window)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)


        print("Se antreneaza modelul CNN")
        for epoch in range(epochs):
            print(f"Se antrenează Epoca {epoch + 1}")
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                # Ne asiguram ca inputul este rgb
                if inputs.dim() == 4:
                    if inputs.shape[1] != 3:   #in caz ca nu este RGB
                        inputs = inputs.repeat(1, 3, 1, 1)

                optimizer.zero_grad()
                outputs = model(inputs)  # model
                loss = criterion(outputs, labels)  # calculez pierderea
                loss.backward()  # backpropagation
                optimizer.step()  # acturalizare
                running_loss += loss.item()
            print(f"Epoca {epoch + 1} din {epochs}, Loss: {running_loss / len(train_loader)}")

            # Validare

            model.eval()
            with torch.no_grad():
                all_inputs = []
                all_labels = []
                print("Prelucrarea Datelor")
                for inputs, labels in train_loader:
                    if inputs.shape[1] == 64:  # pentru cazul [batch_size, 64, 64, 3]
                        inputs = inputs.permute(0, 3, 1, 2)  # [batch_size, 3, 64, 64]
                    all_inputs.append(inputs)
                    all_labels.append(labels)

                all_inputs = torch.cat(all_inputs, dim=0)
                all_labels = torch.cat(all_labels)
                print(f"Dimensiune totala input: {all_inputs.shape}")  # final input dimansions

                outputs = model(all_inputs)
                _, predictii = torch.max(outputs, 1)
                acuratetea_curenta = (predictii == all_labels).float().mean().item()
                print(f"Acuratetea la epoca {epoch + 1} este de {acuratetea_curenta}")

                if acuratetea_curenta > max_acc:
                    max_acc = acuratetea_curenta
                    torch.save(model, cnn_save_file)
                    self.best_model = model

        print(f"Modelul final are acuratetea maxima: {max_acc}")


    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]

        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]

        sorted_indices = np.flipud(np.argsort(image_scores))
        img_detec_sortat = image_detections[sorted_indices]
        scoruri_sortate = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.1
        for i in range(len(img_detec_sortat) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(img_detec_sortat)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(img_detec_sortat[i],img_detec_sortat[j]) > iou_threshold:is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (img_detec_sortat[j][0] + img_detec_sortat[j][2]) / 2
                            c_y = (img_detec_sortat[j][1] + img_detec_sortat[j][3]) / 2
                            if img_detec_sortat[i][0] <= c_x <= img_detec_sortat[i][2] and \
                                    img_detec_sortat[i][1] <= c_y <= img_detec_sortat[i][3]:
                                is_maximal[j] = False
        return img_detec_sortat[is_maximal], scoruri_sortate[is_maximal]

    def run(self):

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = None  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le obtinem
        file_names = np.array([])  # array cu fisierele, in aceasta lista fisierele vor aparea de mai multe ori
        num_test_images = len(test_files)

        model_cnn = self.best_model
        model_cnn.eval()

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i])
            image_scores = []
            image_detections = []

            for scale in self.params.image_scale:
                # print(scale)
                resized_img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
                num_rows, num_cols = resized_img.shape[:2]

                if num_rows >= self.params.dim_window and num_cols >= self.params.dim_window:
                    for y in range(0, num_rows - self.params.dim_window + 1, 8):  # din 8 in 8 pixeli
                        for x in range(0, num_cols - self.params.dim_window + 1, 8):  # din 8 in 8 pixeli
                            patch = resized_img[y:y + self.params.dim_window, x:x + self.params.dim_window]

                            patch_tensor = torch.tensor(patch, dtype=torch.float32)
                            patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(
                                0)
                            patch_tensor = patch_tensor / 255.0  # normalizare

                            with torch.no_grad():
                                output = model_cnn(patch_tensor)
                                output_layer_1 = output[0, 0].item()
                                output_layer_2 = output[0, 1].item()

                                # scorul este diferenta dintre cele 2 layere
                                score = output_layer_2 - output_layer_1
                                # print(score)

                            if score > self.params.threshold:
                                x_min = int(x * (1 / scale))
                                y_min = int(y * (1 / scale))
                                x_max = int((x + self.params.dim_window) * (1 / scale))
                                y_max = int((y + self.params.dim_window) * (1 / scale))

                                image_detections.append([x_min, y_min, x_max, y_max])
                                image_scores.append(score)

            #NMS
            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(np.array(image_detections),
                                                                              np.array(image_scores), img.shape)

            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))

                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'% (i+1, num_test_images, end_time - start_time))

        return np.array(detections), np.array(scores), np.array(file_names)


    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteaza detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()

