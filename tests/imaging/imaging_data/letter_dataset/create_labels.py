import os
import json

CURRENT_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
imgs_path = CURRENT_FILE_PATH + "/12.06.23/images"
labels_path = CURRENT_FILE_PATH + "/12.06.23/labels"
with open(f"{CURRENT_FILE_PATH}/letter_labels.json") as json_file:
    letter_to_labels = json.load(json_file)

for img_file_name in os.listdir(imgs_path):
    name_fragments = img_file_name.split('_')
    letter = name_fragments[1][0]
    label = letter_to_labels[letter]
    with open(f"{labels_path}/{img_file_name.split('.')[0]}.txt", 'w') as f:
        f.write(str(label))
