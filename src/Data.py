import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class Data:
    """ Data class.
    This class is used to load the data from the csv and images files.

    Attributes:
        x_tab (np.ndarray): The csv data samples. For each sample, we have a list of 192 features.
        y_tab (np.ndarray): The csv data labels. For each label, we have a string representing the leaf species.
        x_train (np.ndarray): The csv training data samples.
        x_test (np.ndarray): The csv testing data samples.
        y_train (np.ndarray): The csv training data labels.
        y_test (np.ndarray): The csv testing data labels.
        x_image_tab (np.ndarray): The images data samples. For each sample, we have a matrix of size (width, height).
            - Those samples are related to y_tab.
            - The images doesn't have the same size.
        x_image_train (np.ndarray): The images training data samples.
        x_image_test (np.ndarray): The images testing data samples.
        y_image_train (np.ndarray): The images training data labels.
        y_image_test (np.ndarray): The images testing data labels.
        resized_images_width (int): The width of the resized images.
        resized_images_height (int): The height of the resized images.
    """

    def __init__(self, test_size: float = 0.25, random_state: int = 2, include_images: bool = True):
        """ Constructor for the Data object.

        :param test_size (float): The size of the testing data. Default value is 0.25.
        """
        df: pd.DataFrame = pd.read_csv('src/data/data.csv')

        # Csv data.
        self.x_tab: [np.ndarray] = df.drop(['id', 'species'], axis=1).values
        self.y_tab: [np.ndarray] = df['species'].values

        sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=random_state)
        train = sss.split(self.x_tab, self.y_tab).__next__()
        self.x_train, self.x_test = self.x_tab[train[0]], self.x_tab[train[1]]
        test = sss.split(self.x_tab, self.y_tab).__next__()
        self.y_train, self.y_test = self.y_tab[test[0]], self.y_tab[test[1]]

        if not include_images:
            return
        # Images data.
        self.x_image_tab = [plt.imread('src/data/images/{}.jpg'.format(leaf[0])) for leaf in df.values]

        # List of matrix of size (max_width, max_height).
        resized_images = self.__resize_images()
        self.resized_images_width, self.resized_images_height = resized_images[0].shape[0], resized_images[0].shape[1]
        # Matrix of size(nb_images, max_width * max_height).
        flatten_resized_images = np.array([image.flatten() for image in resized_images])

        sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=random_state)
        train = sss.split(flatten_resized_images, self.y_tab).__next__()
        self.x_image_train, self.x_image_test = flatten_resized_images[train[0]], flatten_resized_images[train[1]]
        test = sss.split(flatten_resized_images, self.y_tab).__next__()
        self.y_image_train, self.y_image_test = self.y_tab[test[0]], self.y_tab[test[1]]

    def __resize_images(self) -> [np.ndarray]:
        """ Resize the images to the max width and height of the images.
        Put the images in the center of the new resized images. The background is black.

        :return: The resized images.
        """
        max_width = np.max([image.shape[0] for image in self.x_image_tab])
        max_height = np.max([image.shape[1] for image in self.x_image_tab])

        resized_images = []
        for image in self.x_image_tab:
            a = np.zeros((max_width, max_height))  # Black background.

            # Put the image on the top left of the new array.
            # a[:image.shape[0], :image.shape[1]] = image

            # Put the image in the center of the new array.
            a[int((max_width - image.shape[0]) / 2):int((max_width + image.shape[0]) / 2),
            int((max_height - image.shape[1]) / 2):int((max_height + image.shape[1]) / 2)] = image

            resized_images.append(a)

        return resized_images
    
    def print_images_from_class(self, class_name: str, nb_images: int = 10):
        """ Print the first nb_images images of the class class_name as a subplot.

        :param class_name (str): The class name.
        :param nb_images (int): The number of images to print. Default value is 10.
        """
        images = [image for image, label in zip(self.x_image_tab, self.y_tab) if label == class_name]
        fig, ax = plt.subplots(1, nb_images, figsize=(20, 20))
        for i in range(nb_images):
            ax[i].imshow(images[i], cmap='gray')
            ax[i].axis('off')
        plt.show()
        
