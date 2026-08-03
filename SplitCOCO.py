import json
from sklearn.model_selection import train_test_split

"""From: https://github.com/Ela-Kan/coco-data-loader/"""

class splitCOCO:
    """Class splits the .json file into test, train and validation .json files 
    using a 70:20:10 (train : test : validation) split."""

    def __init__(self, file_name):
        """Pre-allocate all neccessary variables."""
        self.file_name = file_name
        self.full_dataset = dict()
        self.train = dict()
        self.test = dict()
        self.val = dict()

    def loadCOCO(self):
        """Given a filename to the complete COCO containing all of the data, a dictionary is created from the json information and information about the file is printed
        Args:
            file_name (str): name of file where COCO data is stored
        Returns:
            full_dataset (dictionary): dictionary containing all data."""
        # Load in data from COCO file
        coco_file = open(self.file_name, encoding='utf-8')  # open the specific file
        self.full_dataset = json.load(coco_file)  # read in the whole json information

        print("-------------- Loaded in data information ------------------")
        print('Number of Categories:', len(self.full_dataset['categories']))
        print('Number of Images:', len(self.full_dataset['images']))
        print('Number of Annotations:', len(self.full_dataset['annotations']))
        return self.full_dataset

    def splitData(self):
        """Takes the full_dataset dictionary and splits it into train/test/validation dictionaries in ratio 70:20:10
        Args:
            full_dataset (dict): a complete json coco dataset
        Returns:
            train (dict): training data
            val (dict): validation data
            test (dict): testing data
        """
        #   Split COCO into train/test/validation using 'train_test_split() from sklearn into 70:20:10 (train : test : validation) 
        train_images, temp_images = train_test_split(
            self.full_dataset['images'], test_size=0.3, random_state=13
        )
        # Split temp (30%) into validation (10%) and test (20%)
        val_images, test_images = train_test_split(
            temp_images, test_size=(2/3), random_state=13
        )

        # Create dictionaries with relevant information
        self.train['info'] = self.full_dataset['info']
        self.train['categories'] = self.full_dataset['categories']
        self.train['images'] = train_images
        self.train['annotations'] = self.fetchAnnotations(train_images)  # find corresponding annotations

        self.val['info'] = self.full_dataset['info']
        self.val['categories'] = self.full_dataset['categories']
        self.val['images'] = val_images
        self.val['annotations'] = self.fetchAnnotations(val_images)  # find corresponding annotations

        self.test['info'] = self.full_dataset['info']
        self.test['categories'] = self.full_dataset['categories']
        self.test['images'] = test_images
        self.test['annotations'] = self.fetchAnnotations(test_images)  # find corresponding annotations

        # Print info
        print('Number of Training images (before augmentation):', len(train_images))
        print('Number of Validation images:', len(val_images))
        print('Number of Testing images:', len(test_images))
        return self.train, self.test, self.val


    def fetchAnnotations(self, images):
        """Given the full_dataset (the json), and a list of image information, find the corresponding annotations to the selected images
        Args:
            full_dataset (dict): a complete json coco dataset
            images (list): image information from desired split (e.g. test/train/val)
        Returns:
            annotations_list (list): annotations corresponding to the given images
        """
        image_ids = {img['id'] for img in images}
        annotations_list = []
        for annotation in self.full_dataset['annotations']:
            if annotation['image_id'] in image_ids:
                annotations_list.append(annotation)
        return annotations_list

    def saveTrainTestValidationCOCO(self):
            """Saves train, test and validation dictionaries into individual COCO .json files
            Args:
                train (dict): training data
                val (dict): validation data
                test (dict): testing data
                """
            # Create new json files with the information from the above split by creating dictionaries for test, train and validation and save these
            with open('coco_train.json', 'w', encoding='utf-8') as train_file:  # write training data to file
                json.dump(self.train, train_file, ensure_ascii=False)

            with open('coco_val.json', 'w', encoding='utf-8') as val_file:  # write training data to file
                json.dump(self.val, val_file, ensure_ascii=False)

            with open('coco_test.json', 'w', encoding='utf-8') as test_file:  # write training data to file
                 json.dump(self.test, test_file, ensure_ascii=False)
            
    def run(self):
        self.loadCOCO()  # load COCO data
        self.train, self.test, self.val = self.splitData()  # create test/train/val split
        # save split COCO data into new .json files
        self.saveTrainTestValidationCOCO()


if __name__ == "__main__":
    splitCOCO('coco.json').run()