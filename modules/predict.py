# <YOUR_IMPORTS>
import logging
from datetime import datetime

import dill
import os
import json
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')
folder_path = f'{path}/data/test'


def predict() -> None:
    model = load_model()

    json_files = load_test()

    list = []
    for json_file in json_files:
        file = os.path.join(folder_path, json_file)
        file_name = json_file.split('.json')
        with open(file, 'r') as f:
            data = [json.load(f)]
            dict = {}
            df = pd.DataFrame(data)
            prediction = model.predict(df)
            dict['car_id'] = file_name[0]
            dict['pred'] = prediction[0]

            list.append(dict)

    pd.DataFrame.from_dict(list).to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    logging.info('Data were saved')


def load_model():
    with open(get_latest_path_model(), 'rb') as file:
        return dill.load(file)


def get_latest_path_model():
    directory = f'{path}/data/models'
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def load_test():
    return [f for f in os.listdir(folder_path) if f.endswith('.json')]


if __name__ == '__main__':
    predict()
