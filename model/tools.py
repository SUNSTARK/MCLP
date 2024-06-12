import os
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

from dataloader import MyDataset
from framework import MyModel


def get_config(path, easy):
    """
    get config yml
    :param easy: is easydict mode
    :param path: yml path
    :return: EasyDict format
    """
    f = open(path, 'r', encoding='utf-8')
    res = yaml.safe_load(f)
    if easy:
        return EasyDict(res)
    else:
        return res


def get_user_input():
    while True:
        user_input = input("Configuration yml file modification completed? (y/n):").strip().lower()
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False
        else:
            print("Invalid Options")

def custom_collate(batch, device, config):
    batch_dict = {
        'user': torch.tensor([item['user'] for item in batch]).to(device),
        'location_x': torch.stack([torch.tensor(item['location_x']) for item in batch]).to(device),
        'hour': torch.stack([torch.tensor(item['hour']) for item in batch]).to(device),
        'location_y': torch.tensor([item['location_y'] for item in batch]).to(device),
        'timeslot_y': torch.tensor([item['timeslot_y'] for item in batch]).to(device),
        'hour_mask': torch.stack([torch.tensor(item['hour_mask']) for item in batch]).to(device),
        'prob_matrix_time_individual': torch.stack([torch.tensor(item['prob_matrix_time_individual']) for item in batch]).to(device),
    }

    if config.Dataset.topic_num > 0:
        batch_dict['user_topic_loc'] = torch.stack([torch.tensor(item['user_topic_loc'], dtype=torch.float32) for item in batch]).to(device)

    return batch_dict

def update_config(path, key_list, value):
    """
    update config yml
    :param key_list: yml key list
    :param path: yml path
    :param value: corresponding value
    :return:
    """
    config = get_config(path, easy=False)

    current_level = config
    outer_key = key_list[0]
    inner_key = key_list[1]
    if outer_key not in current_level:
        print(f'Update config Error: outermost key {outer_key} not exist!')
        exit()
    if inner_key not in current_level[outer_key]:
        print(f'Update config Error: inner key {inner_key} not exist in {outer_key}!')
        exit()

    current_level[outer_key][inner_key] = value

    with open(path, 'w') as f_writer:
        yaml.dump(config, f_writer, default_flow_style=False)
        f_writer.close()


def get_mapper(dataset_path):
    location_mapper_path = os.path.join(dataset_path, 'location_mapper.npy')
    user_mapper_path = os.path.join(dataset_path, 'user_mapper.npy')

    if os.path.exists(location_mapper_path) and os.path.exists(user_mapper_path):
        return

    location_set = set()
    user_set = set()

    with open(os.path.join(dataset_path, 'train.csv'), encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.strip().split(',')
            uid = elements[0]
            item_seq = elements[1:]

            user_set.add(uid)

            for item in item_seq:
                loc = item.split('@')[0]
                location_set.add(loc)
        f.close()
    with open(os.path.join(dataset_path, 'test.csv'), encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.strip().split(',')
            uid = elements[0]
            item_seq = elements[1:]

            user_set.add(uid)

            for item in item_seq:
                loc = item.split('@')[0]
                location_set.add(loc)
        f.close()

    location2id = {location: idx for idx, location in enumerate(location_set)}
    user2id = {user: idx for idx, user in enumerate(user_set)}

    print('\n*** Please check the corresponding dataset configuration yml file. ***')
    print('unique location num:', len(location2id))
    print('unique user num:', len(user2id))

    yml_modified = get_user_input()
    if yml_modified:
        np.save(location_mapper_path, location2id)
        np.save(user_mapper_path, user2id)
    else:
        print('Program Exit')
        exit()


def run_test(dataset_path, model_path, model, device, epoch, test_only):
    config_path = os.path.join(model_path, f"settings.yml")

    config = get_config(config_path, easy=True)
    dataset = MyDataset(config=config, dataset_path=dataset_path, device=device, load_mode='test')

    batch_size = config.Model.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, collate_fn=lambda batch: custom_collate(batch, device, config))
    print('Test batches:', len(dataloader))

    if test_only:
        saved_model_path = os.path.join(model_path, f'model_checkpoint_epoch{epoch}.pth')
        model.load_state_dict(torch.load(saved_model_path, map_location=device)['model_state_dict'])
        model.to(device)

    model.eval()
    precision_loc = 0
    top_k_values = [1, 3, 5, 10]
    top_k_correct_loc = np.array([0 for _ in range(len(top_k_values))])
    total_samples = 0

    def evaluate(output, label, ks):
        topk_correct_counts = [
            torch.sum(
                (torch.topk(output, k=top_k, dim=1)[1] + 0) == label.unsqueeze(1)
            ).item()
            for top_k in ks
        ]
        return np.array(topk_correct_counts)

    def calculate_mrr(output, true_labels):
        res = 0.0
        for i, pred in enumerate(output):
            sorted_indices = torch.argsort(pred, descending=True)
            true_index = np.where(true_labels[i].cpu() == sorted_indices.cpu())[0]
            if len(true_index) > 0:
                res += 1.0 / (true_index[0] + 1)
        return res

    with torch.no_grad():
        for batch_data in dataloader:
            location_output = model(batch_data)
            location_y = batch_data['location_y']
            location_y = location_y.view(-1)
            total_samples += location_y.size(0)

            top_k_correct_loc += evaluate(location_output, location_y, top_k_values)
            precision_loc += calculate_mrr(location_output, location_y)

    top_k_accuracy_loc = [count / total_samples * 100 for count in list(top_k_correct_loc)]
    result_str = "*********************** Test ***********************\n"
    result_str += f"base_dim: {config.Embedding.base_dim} | dim: {config.Embedding.base_dim}\n"
    result_str += f"AT_type: {config.Model.at_type} | topic_num: {config.Dataset.topic_num}\n"
    result_str += f"encoder: {config.Encoder.encoder_type}\n"
    result_str += f"Epoch {epoch + 1}: Total {total_samples} predictions on Next Location:\n"
    for k, accuracy in zip(top_k_values, top_k_accuracy_loc):
        result_str += f"Acc@{k}: {accuracy:.2f}\n"
    result_str += f"MRR: {precision_loc * 100 / total_samples:.2f}"
    result_save = top_k_accuracy_loc
    result_save.append(precision_loc * 100 / total_samples)
    result_save = np.array(result_save)
    np.save(
        f"{model_path}/acc_{epoch + 1}_topic{config.Dataset.topic_num}"
        f"_at{config.Model.at_type}"
        f"_dim{config.Embedding.base_dim}_encoder{config.Encoder.encoder_type}"
        f"_seed{config.Model.seed}.npy",
        result_save)

    print(result_str)

    with open(os.path.join(model_path, 'results.txt'), 'a', encoding='utf8') as res_file:
        res_file.write(result_str + '\n\n')


def train_epoch(model, dataloader, optimizer, loss_fn, scheduler):
    model.train()
    total_loss_epoch = 0.0

    for batch_data in dataloader:
        location_output = model(batch_data)
        location_y = batch_data['location_y'].view(-1)
        location_loss = loss_fn(location_output, location_y)
        total_loss = location_loss.sum()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()
        total_loss_epoch += total_loss.item()

    return total_loss_epoch / len(dataloader)


def save_checkpoint(save_dir, model, optimizer, best_val_loss, epoch):
    save_path = os.path.join(save_dir, f"model_checkpoint_epoch{epoch}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, save_path)


def get_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
