import os
from sklearn.model_selection import train_test_split
import shutil

images = []
annotations = []

image_path = 'all_images'

label_path = 'all_labels'


# Read images and annotations
for x in os.listdir(image_path):
    pathDir = os.path.join(image_path,x)
    images.append(pathDir)

for x in os.listdir(label_path):
    pathDir = os.path.join(label_path,x)
    annotations.append(pathDir)

images.sort()
annotations.sort()

train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.1, random_state = 1)
# val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, shuffle=True, random_state = 1)

#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)

# Move the splits into their folders
move_files_to_folder(train_images, 'BaleDataset/images/train')
move_files_to_folder(val_images, 'BaleDataset/images/validate')

move_files_to_folder(train_annotations, 'BaleDataset/labels/train')
move_files_to_folder(val_annotations, 'BaleDataset/labels/validate')
