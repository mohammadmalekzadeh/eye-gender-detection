### importing library
import pandas as pd
import numpy as np
import os

### read image path
male_image = os.listdir('../data/images/maleeyes/')
female_image = os.listdir('../data/images/femaleeyes/')
path = os.listdir('../data/images/')

### labeling and save labels
currect_gender = 'male'
currect_path = '../data/images/maleeyes/'
data = []
for gender in [male_image, female_image]:
    for image in gender:
        data.append({'image_path': currect_path + image, 'gender': currect_gender})
    currect_gender = 'female'
    currect_path = '../data/images/femaleeyes/'
pd.DataFrame(data).to_csv('../data/labels.csv', index=False)