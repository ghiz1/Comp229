import numpy as np
import albumentations as A
from tensorflow.keras.utils import Sequence

def calculate_label_distribution(df, label_column):
    label_counts = df[label_column].value_counts().to_dict()
    return label_counts

class ElasticAugmentationGenerator(Sequence):
    def __init__(self, images, labels, batch_size, transform, label_counts, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform
        self.label_counts = label_counts
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        self.max_class_size = max(label_counts.values())

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.images[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        augmented_images = []
        for image, label in zip(batch_images, batch_labels):
            augmented = image
            if self.label_counts[label] < self.max_class_size:
                augmented = self.transform(image=image)['image']
            augmented_images.append(augmented)
        augmented_images = np.array(augmented_images).astype(np.float32)
        return augmented_images, np.array(batch_labels)

transform = A.Compose([A.ElasticTransform(alpha=1, sigma=50, p=1)])
