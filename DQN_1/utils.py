import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from collections import OrderedDict
import multiprocessing as mp
# from tqdm.notebook import tqdm


def decode_code_to_idx_sparse_vec(state):  # array(6*5*15)

    def find_1_in_array(array):
        for idx, num in enumerate(array):
            if num == 1: return idx
        return 0

    sparse_vec = []
    for idx in range(len(state)):
        cards = []
        for card in state[idx].T:
            cards.append(find_1_in_array(card))

        sparse_vec.append(cards)

    return sparse_vec


def decode_choice_to_one_hot(choices):
    one_hot = [0] * 309
    for choice in choices:
        one_hot[choice] = 1

    return one_hot


f = open('data/action_space.json')
y = open('data/specific_map.json')
action_space = json.load(f, object_pairs_hook=OrderedDict)
specific_map = json.load(y, object_pairs_hook=OrderedDict)


def get_steps(data, game_id, player_id, lamb=0.9, n_reward=1):
    step_list = []
    player = data[game_id][player_id]
    if player:
        total_steps = len(player)
        final_reward = player[-1][2]
        for num_step, step in enumerate(player): # 五元组
            obs_array = decode_code_to_idx_sparse_vec(step[0]['obs'])
            # action_vec = decode_choice_to_one_hot(step[0]['legal_actions'])
            true_choice = action_space[specific_map[step[1]][0]]
            step_list.append((obs_array, 0, true_choice,
                              np.power(lamb, total_steps - num_step - 1) * final_reward * n_reward))
    return step_list


def gen_data_list(data):
    pool = mp.Pool(mp.cpu_count())
    num_split = np.arange(len(data))
    lower = [low[0] for low in np.array_split(num_split, mp.cpu_count())]
    upper = [up[-1] for up in np.array_split(num_split, mp.cpu_count())]
    results = []
    for low, up in zip(lower, upper):
        result = pool.apply_async(get_list_on_one_cpu, args=(data[low:up],))
        results.append(result)

    pool.close()
    pool.join()
    
    data_list = []
    for result in results:
        data_list.extend(result.get())
    return data_list


def get_list_on_one_cpu(data):
#     print('process is going on!')
    
    result = []
    num_games = len(data)
    num_players = 3
    for game in range(num_games):
        for player in range(num_players):
            steps = get_steps(data, game, player)
            for step_idx in range(len(steps)):
                step = steps[step_idx]
                if step:
                    result.append(step)
#     print('process finish!')
    return result
    

class auto_encoder_dataset(Dataset):  # see all games all players the same
    def __init__(self, data_list):
        # self.data = np.load(data_dir, allow_pickle=True)
        # self.num_games = len(self.data)
        # self.num_players = 3
        self.data_list = data_list
        # self.gen_data_list()

    def __getitem__(self, idx):
        (state, action, true_action, reward) = self.data_list[idx]
        current_hand = torch.tensor(state[0], dtype=torch.long)
        last_action = torch.tensor(state[2], dtype=torch.long)
        others_hand = torch.tensor(state[1], dtype=torch.long)
        # last_three_action = torch.tensor(state[2:5], dtype=torch.long)
        played_cards = torch.tensor(state[5], dtype=torch.long)
        actions = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        true_action = torch.tensor(true_action, dtype=torch.long)
        return {'current_hand': current_hand,
                'last_action': last_action,
                'others_hand': others_hand,
                # 'last_three_action': last_three_action,
                'played_cards': played_cards,
                'actions': actions,
                'true_action': true_action,
                'reward': reward}

    def __len__(self):
        return len(self.data_list)


def gen_data_loader(data_dir):
    data = np.load(data_dir,allow_pickle=True)
    data_list = gen_data_list(data)
    dataset = auto_encoder_dataset(data_list)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
    return data_loader

