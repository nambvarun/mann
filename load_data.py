import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
        shuffle: Shuffle the examples.
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = misc.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
            batch_size: batch size
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        all_image_batches = np.zeros((batch_size, self.num_samples_per_class, self.num_classes, 784))
        all_label_batches = np.zeros((batch_size, self.num_samples_per_class, self.num_classes, self.num_classes))

        for batch_idx in range(batch_size):
            _shuffle = True

            # sample with replacement per batch.
            # sample n different classes from different characters.
            classes_idx = np.random.choice(len(folders), self.num_classes, replace=False)
            paths_to_sample = [folders[i] for i in classes_idx.tolist()]

            # sample k examples of each character.
            one_hot_mat = np.eye(self.num_classes)
            label_path_mapping = get_images(paths_to_sample, one_hot_mat, nb_samples=self.num_samples_per_class, shuffle=_shuffle)

            one_hot_to_class_map = {i: self.num_samples_per_class-1 for i in range(self.num_classes)}

            for one_hot_label, img_path in label_path_mapping:
                # get image
                img_np = image_file_to_array(img_path, 784)

                # input [batch idx, sample idx, class idx, dat]
                n = np.where(one_hot_label == 1)[0][0]
                k = one_hot_to_class_map[n]

                all_image_batches[batch_idx, k, n, :] = img_np
                all_label_batches[batch_idx, k, n, :] = one_hot_label

                one_hot_to_class_map[n] -= 1

            if _shuffle:
                all_image_batches_shuffled = np.zeros_like(all_image_batches[batch_idx, :, :, :])
                all_label_batches_shuffled = np.zeros_like(all_label_batches[batch_idx, :, :, :])

                for train_example in range(self.num_samples_per_class):
                    new_order = np.random.choice(self.num_classes, self.num_classes, replace=False).tolist()

                    for unshuffled_ind, shuffled_ind in enumerate(new_order):
                        all_image_batches_shuffled[train_example, shuffled_ind, :] = all_image_batches[batch_idx, train_example, unshuffled_ind, :]
                        all_label_batches_shuffled[train_example, shuffled_ind, :] = all_label_batches[batch_idx, train_example, unshuffled_ind, :]

                all_image_batches[batch_idx, :, :, :] = all_image_batches_shuffled
                all_label_batches[batch_idx, :, :, :] = all_label_batches_shuffled
        #############################

        return all_image_batches, all_label_batches

dg = DataGenerator(3, 2)
dg.sample_batch('test', 1)
