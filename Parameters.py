import os

class Parameters:
    def __init__(self):
        self.base_dir = '../..'
        self.dir_pos_examples = os.path.join(self.base_dir, 'auxiliare/64x64/all')
        print(self.dir_pos_examples)
        self.dir_neg_examples = os.path.join(self.base_dir, 'auxiliare/64x64_negative_3')
        print(self.dir_pos_examples)
        self.dir_test_examples = os.path.join(self.base_dir,'testare_t3')  # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.path_annotations = os.path.join(self.base_dir, 'validare/task1_gt_validare2.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'CNN/dataset/salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 64  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 8  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 6713  # numarul exemplelor pozitive
        self.number_negative_examples = 10000  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0
        self.image_scale = [1.0]
        self.use_flip_images = None

        #cnn
        self.epochs = 20
