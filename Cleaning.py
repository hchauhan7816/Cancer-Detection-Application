import pandas as pd
from skimage import io
import os
from pathlib import Path

# dataset link :- https://www.kaggle.com/datasets/ashery/chexpert

chest_xray_df = pd.read_csv('CheXpert-v1.0-small/train.csv')

print(f"CancerDetected: {(chest_xray_df['Lung Lesion'] == 1).sum()}\n \
      Unreliable/Unsure(labelled 0): {(chest_xray_df['Lung Lesion'] == 0).sum()}\n \
      No Cancer(labelled -1): {(chest_xray_df['Lung Lesion'] == -1).sum()}")

i = 0
cancer_images = chest_xray_df[chest_xray_df['Lung Lesion'] == 1]
cancer_images = cancer_images[['Path', 'Lung Lesion']]
image_path = cancer_images.iloc[i]['Path'].split('/')
'_'.join([x for x in image_path[-3:]])


directory = "malignantt"
parent_dir = "data/train/"
path = os.path.join(parent_dir, directory)
os.mkdir(path)

new_image_root_dir = 'data/train/malignantt/'

for i in range(len(cancer_images)):
    image_path = cancer_images.iloc[i]['Path'].split('/')

    new_image_name = new_image_root_dir + \
        '_'.join([x for x in image_path[-3:]])
    im = io.imread(cancer_images.iloc[i]['Path'])
    io.imsave(new_image_name, im)


no_cancer_images = chest_xray_df[chest_xray_df['Lung Lesion'] == -1]
no_cancer_images = no_cancer_images[['Path', 'Lung Lesion']]
len(no_cancer_images)


new_image_root_dir = 'data/train/benign/'

for i in range(len(no_cancer_images)):
    image_path = no_cancer_images.iloc[i]['Path'].split('/')
    new_image_name = new_image_root_dir + \
        '_'.join([x for x in image_path[-3:]])

    im = io.imread(no_cancer_images.iloc[i]['Path'])
    io.imsave(new_image_name, im)


chest_xray_df = pd.read_csv('CheXpert-v1.0-small/train.csv')
print(f"NoFindings: {(chest_xray_df['No Finding'] == 1).sum()}\n")
no_findings_df = chest_xray_df[chest_xray_df['No Finding'] == 1]
no_findings_df = no_findings_df[['Path']]
len(no_findings_df)


def move_image_in_specific_row_to_folder(no_findings_df, i, image_root_dir):
    image_path = no_findings_df.iloc[i]['Path'].split('/')
    new_image_name = image_root_dir + '_'.join([x for x in image_path[-3:]])

    im = io.imread(no_findings_df.iloc[i]['Path'])
    io.imsave(new_image_name, im)


i = 0
image_root_dir = 'data/train/benign/'
while i < 5000:
    # Move to train/benign
    move_image_in_specific_row_to_folder(no_findings_df, i, image_root_dir)
    i += 1

image_root_dir = 'data/val/benign/'
while i < 6500:
    # Move to val/benign
    move_image_in_specific_row_to_folder(no_findings_df, i, image_root_dir)
    i += 1

image_root_dir = 'data/test/benign/'
while i < 8000:
    # Move to test/benign
    move_image_in_specific_row_to_folder(no_findings_df, i, image_root_dir)
    i += 1


i = 0
image_root_dir = 'data/train/malignant/'
while i < 6286:
    # Move to train/malignant
    move_image_in_specific_row_to_folder(cancer_images, i, image_root_dir)
    i += 1

image_root_dir = 'data/val/malignant/'
while i < 7736:
    # Move to val/malignant
    move_image_in_specific_row_to_folder(cancer_images, i, image_root_dir)
    i += 1

image_root_dir = 'data/test/malignant/'
while i < 9186:
    # Move to test/malignant
    move_image_in_specific_row_to_folder(cancer_images, i, image_root_dir)
    i += 1


directory = "malignantt"
parent = "C:/Users/jatin/cancer_detection/data/train/"
path = os.path.join(parent, directory)
os.rmdir(path)
