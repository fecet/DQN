import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0')


class sequence_encoder(nn.Module):
    def __init__(self):
        super(sequence_encoder, self).__init__()
        self.embedding_layer = nn.Embedding(5, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(512, 8, 1024)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 5)

    def forward(self, embedding_list):
        embeddings = []
        for embedding in embedding_list:
            embeddings.append(self.embedding_layer(embedding))
        out = torch.zeros(embeddings[0].shape).to(device)
        for i in embeddings:
            out += i
        out = self.encoder(out.permute(1, 0, 2)).permute(1, 0, 2)
        return out


class calculate_q_value_dqn(nn.Module):
    def __init__(self):
        super(calculate_q_value_dqn, self).__init__()
        self.encoder = sequence_encoder()
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 376)
        self.q_value_loss = nn.MSELoss()

    def forward(self, state_dict):
        current_hand = state_dict['current_hand'].to(device)
        last_action = state_dict['last_action'].to(device)
        others_hand = state_dict['others_hand'].to(device)
        played_cards = state_dict['played_cards'].to(device)
        # actions = state_dict['actions'].to(device)
        true_action = state_dict['true_action'].to(device)
        reward = state_dict['reward'].to(device)

        embedding_list = [current_hand, last_action, others_hand, played_cards]
        embeddings = self.encoder(embedding_list)  # (batch_size, length(13), embedding_features(512))

        out = torch.tanh(self.linear1(embeddings[:, 0, :]))
        out = self.linear2(out)
        label = torch.tensor([[i.item()] for i in true_action], dtype=torch.long, device=device)
        q_values = torch.gather(out, 1, label).squeeze()
        loss = self.q_value_loss(q_values, reward)
        return loss











