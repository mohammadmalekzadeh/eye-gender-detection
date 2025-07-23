### definition function
def label_creator() -> None:
    from src.utlis import BASE_DIR
    ### importing library
    import pandas as pd
    import numpy as np
    import os

    ### read image path
    male_image = os.listdir(BASE_DIR+'/data/images/maleeyes/')
    female_image = os.listdir(BASE_DIR+'/data/images/femaleeyes/')
    path = os.listdir(BASE_DIR+'/data/images/')

    ### labeling and save labels
    currect_gender = 'male'
    currect_path = '../data/images/maleeyes/'
    data = []
    for gender in [male_image, female_image]:
        for image in gender:
            data.append({'image_path': currect_path + image, 'gender': currect_gender})
        currect_gender = 'female'
        currect_path = '../data/images/femaleeyes/'
    df = pd.DataFrame(data)
    df['num'] = df['image_path'].apply(lambda x : int(x.split('/')[-1].split('.')[0]))
    df.sort_values(by='num', ascending=True, inplace=True)
    df.drop(columns='num', inplace=True)
    df.to_csv(BASE_DIR+'/data/labels.csv', index=False)
    return "[!] Images have been successfully labeled"