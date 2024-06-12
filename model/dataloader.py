import datetime
import os

import gensim
import numpy as np
import torch
from gensim import models
from torch.utils.data import Dataset
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, config, dataset_path, device, load_mode):
        self.config = config
        self.device = device
        self.load_mode = load_mode
        self.dataset_path = dataset_path

        self.user2id = np.load(os.path.join(dataset_path, 'user_mapper.npy'), allow_pickle=True).item()
        self.location2id = np.load(os.path.join(dataset_path, 'location_mapper.npy'), allow_pickle=True).item()

        if self.config.Dataset.topic_num > 0:
            if not os.path.exists(os.path.join(dataset_path, f'user_topic_loc_{self.config.Dataset.topic_num}.npy')):
                self.preprocess_data(topic_num=self.config.Dataset.topic_num)

        if load_mode == 'test':
            self.data = self.load_npy_file(os.path.join(dataset_path, f'{load_mode}.npy'))
        else:
            if not os.path.exists(os.path.join(dataset_path, f'{load_mode}.npy')):
                self.preprocess_data(topic_num=0)
                self.generate_data(load_mode='train')
                self.generate_data(load_mode='test')
            self.data = self.load_npy_file(os.path.join(dataset_path, f'{load_mode}.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        return data

    def preprocess_data(self, topic_num):
        trans_time_individual = []
        occur_time_individual = np.zeros(shape=(len(self.user2id), 24), dtype=np.float32)
        user_loc_matrix = np.zeros((len(self.user2id), len(self.location2id)))  # for LDA

        diff_data = []
        with open(os.path.join(self.dataset_path, f'train.csv'), 'r', encoding='utf8') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc=f'Preprocess data'):
                trans_matrix_time = np.ones((24, 24))
                stay_points = line.strip().split(',')[1:]
                user = line.strip().split(',')[0]
                for i in range(len(stay_points) - 1):
                    location, timestamp = stay_points[i].split('@')
                    next_location, next_timestamp = stay_points[i + 1].split('@')
                    weekday, hour = datetime_to_features(timestamp)
                    next_weekday, next_hour = datetime_to_features(next_timestamp)
                    diff_data.append(abs(next_hour - hour))
                    i, j = hour, next_hour
                    trans_matrix_time[i, j] += 1
                    occur_time_individual[self.user2id[user]][hour] += 1
                    user_loc_matrix[self.user2id[user], self.location2id[location]] += 1
                    if i == len(stay_points) - 2:
                        occur_time_individual[self.user2id[user]][next_hour] += 1
                        user_loc_matrix[self.user2id[user], self.location2id[next_location]] += 1

                time_row_sums = trans_matrix_time.sum(axis=1)
                trans_matrix_time = trans_matrix_time / time_row_sums[:, np.newaxis]
                trans_time_individual.append(trans_matrix_time)

        trans_time_individual = np.array(trans_time_individual)

        num_users, num_locations = user_loc_matrix.shape
        dictionary = gensim.corpora.Dictionary([[str(i)] for i in range(num_locations)])
        corpus = []
        for user in user_loc_matrix:
            user_doc = [str(loc) for loc, count in enumerate(user) for _ in range(int(count))]
            corpus.append(dictionary.doc2bow(user_doc))
        if topic_num > 0:
            print(f'Generating a probability distribution... topic: {topic_num}')
            lda = models.LdaModel(corpus, num_topics=topic_num, random_state=42)
            user_topics = np.zeros((num_users, topic_num))
            for i, user in enumerate(user_loc_matrix):
                user_doc = [str(loc) for loc, count in enumerate(user) for _ in range(int(count))]
                for item in lda[dictionary.doc2bow(user_doc)]:
                    j = item[0]
                    prob = item[1]
                    user_topics[i, j] = prob
            np.save(os.path.join(self.dataset_path, f'user_topic_loc_{topic_num}.npy'),
                    np.array(user_topics))

        np.save(os.path.join(self.dataset_path, f'prob_matrix_time_individual.npy'),
                np.array(trans_time_individual))
        np.save(os.path.join(self.dataset_path, f'occur_time_individual.npy'),
                np.array(occur_time_individual))

    def generate_data(self, load_mode):
        occur_time_individual = np.load(os.path.join(self.dataset_path, f'occur_time_individual.npy'),
                                        allow_pickle=True)
        res = []
        with open(os.path.join(self.dataset_path, f'{load_mode}.csv'), 'r', encoding='utf8') as file:
            lines = file.readlines()
            for line_i, line in enumerate(tqdm(lines, desc=f'Initial {load_mode} data')):
                user = line.strip().split(',')[0]
                occur_time_user = occur_time_individual[self.user2id[user]]
                stay_points = line.strip().split(',')[1:]
                sequence_count, left = divmod(len(stay_points), self.config.Dataset.sequence_length)
                assert sequence_count > 0, f"{user}'s does not have enough data."
                sequence_count -= 1 if left == 0 else 0
                for i in range(sequence_count):
                    split_start = i * self.config.Dataset.sequence_length
                    split_end = (i + 1) * self.config.Dataset.sequence_length
                    location_x = [self.location2id[item.split('@')[0]] for item in stay_points[split_start:split_end]]
                    timestamp_x = [item.split('@')[1] for item in stay_points[split_start:split_end]]
                    location_y = [self.location2id[item.split('@')[0]] for item in
                                  stay_points[split_start + 1:split_end + 1]]
                    timestamp_y = [item.split('@')[1] for item in stay_points[split_start + 1:split_end + 1]]
                    timeslot_y = []
                    hour_x = []
                    hour_mask = []
                    for item in timestamp_x:
                        weekday, hour = datetime_to_features(item)
                        hour_x.append(hour)
                        mask = np.zeros(24, dtype=np.int32)
                        mask[occur_time_user == 0] = 1
                        if mask.sum() == 24:
                            exit()
                        hour_mask.append(mask)
                    for item in timestamp_y:
                        weekday, hour = datetime_to_features(item)
                        timeslot_y.append(hour)
                    res.append(
                        {
                            'user': self.user2id[user],
                            'location_x': location_x,
                            'hour': hour_x,
                            'hour_mask': np.array(hour_mask),
                            'location_y': location_y,
                            'timeslot_y': timeslot_y,
                        }
                    )

        np.save(os.path.join(self.dataset_path, f'{load_mode}.npy'), res)

    def load_npy_file(self, save_path):
        loaded_data = np.load(save_path, allow_pickle=True)
        prob_matrix_time_individual = np.load(
            os.path.join(self.dataset_path, f'prob_matrix_time_individual.npy'),
            allow_pickle=True)
        for data in loaded_data:
            user_idx = data['user']
            data['prob_matrix_time_individual'] = prob_matrix_time_individual[user_idx]
        if self.config.Dataset.topic_num > 0:
            user_topic_loc = np.load(
                os.path.join(self.dataset_path, f'user_topic_loc_{self.config.Dataset.topic_num}.npy'),
                allow_pickle=True)
            for data in loaded_data:
                user_idx = data['user']
                data['user_topic_loc'] = user_topic_loc[user_idx]
        return loaded_data


def datetime_to_features(timestamp):
    dt = datetime.datetime.fromtimestamp(int(timestamp) // 1000)
    weekday = dt.weekday()
    hour = dt.hour
    return weekday, hour
