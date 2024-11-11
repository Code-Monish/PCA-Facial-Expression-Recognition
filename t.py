import os
class datasetClass:

    def __init__(self, required_no):
        dataset_name = "faces94"
        dir = "PCA-Facial-Expression-Recognition/" + dataset_name

        self.images_path_for_training = []
        self.labels_for_training = []
        self.no_of_images_for_training = []

        self.images_path_for_testing = []
        self.labels_for_testing = []
        self.no_of_images_for_testing = []

        self.images_target = []       

        per_no = 0
        for name in os.listdir(dir):
            dir_path = os.path.join(dir, name)
            if os.path.isdir(dir_path):
                if len(os.listdir(dir_path)) >= required_no:
                    i = 0
                    for img_name in os.listdir(dir_path):
                        img_path = os.path.join(dir_path, img_name)

                        if i < required_no:
                            self.images_path_for_training += [img_path]
                            self.labels_for_training += [per_no]

                            if len(self.no_of_images_for_training) > per_no:
                                self.no_of_images_for_training[per_no] += 1
                            else:
                                self.no_of_images_for_training += [1]

                            if i == 0:
                                self.images_target += [name]
                        else:
                            self.images_path_for_testing += [img_path]
                            self.labels_for_testing += [per_no]

                            if len(self.no_of_images_for_testing) > per_no:
                                self.no_of_images_for_testing[per_no] += 1
                            
                            else:
                                self.no_of_images_for_testing += [1]
                        
                        i += 1
                    per_no += 1
                            


