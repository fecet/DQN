import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0')


class bce_loss_with_weight(nn.Module):
    def __init__(self, alpha=0.99):
        super(bce_loss_with_weight, self).__init__()
        self.alpha = alpha

    def forward(self, x, label):  # (batch_size, num_labels) full of 1 0
        x = torch.sigmoid(x)
        loss = - (1 - self.alpha) * (1 - label) * torch.log(1 - x) - self.alpha * label * torch.log(x)
        loss = loss.mean()
        return loss


class sequence_state_encoder(nn.Module):
    def __init__(self):
        super(sequence_state_encoder, self).__init__()
        self.embedding_layer = nn.Embedding(5, 512)
        self.encoder_layer = nn.TransformerEncoderLayer(512, 8, 1024)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, 3)

    def forward(self, input_seq):
        embeddings = self.embedding_layer(input_seq)
        embedding_for_encoder = embeddings.permute(1, 0, 2)
        out = self.encoder(embedding_for_encoder)
        out = out.permute(1, 0, 2)
        return out  # (batch_size, length(13), embedding_features(512))


class three_task_dqn(nn.Module):
    def __init__(self, n1=1, n2=0, n3=0):
        super(three_task_dqn, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.encoder = sequence_state_encoder()

        # for find_action_task
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 309)
        self.find_action_task_loss = bce_loss_with_weight()

        # for predict_best_action_task
        self.linear3 = nn.Linear(309, 309)
        self.predict_best_action_loss = nn.CrossEntropyLoss()

        # for calculate_q_value_task
        self.linear4 = nn.Linear(512, 309)
        self.calculate_q_value_loss = nn.MSELoss()

    def find_action_task(self, current_hand, last_action, actions):
        state_embeddings = self.encoder(current_hand) + self.encoder(last_action)
        out = self.linear2(torch.tanh(self.linear1(state_embeddings[:, 0, :])))
        loss = self.find_action_task_loss(out, actions)
        return loss

    def predict_best_action(self, current_hand, last_action, played_cards,
                            others_hand, true_action, reward):
        state_embeddings = self.encoder(current_hand) + self.encoder(last_action)
        others_embeddings = self.encoder(played_cards) + self.encoder(others_hand)
        state_out = self.linear2(torch.tanh(self.linear1(state_embeddings[:, 0, :])))
        others_out = self.linear2(torch.tanh(self.linear1(others_embeddings[:, 0, :])))
        out = state_out + others_out
        out = F.relu(self.linear3(out))
        predict_loss = self.predict_best_action_loss(out, true_action)

        # then calculate q_value
        label = torch.tensor([[i.item()] for i in true_action], dtype=torch.long, device=device)
        q_value_out = torch.gather(out, 1, label).squeeze()
        q_value_loss = self.calculate_q_value_loss(q_value_out, reward)
        return predict_loss, q_value_loss

    # def calculate_q_value(self, current_hand, last_action, played_cards,
    #                       others_hand, true_action, reward):
    #     state_embeddings = self.encoder(current_hand) + self.encoder(last_action)
    #     others_embeddings = self.encoder(played_cards) + self.encoder(others_hand)
    #     state_out = self.linear1(state_embeddings[:, 0, :])
    #     others_out = self.linear4(others_embeddings[:, 0, :])
    #     out = state_out + others_out
    #     label = torch.tensor([[i] for i in true_action], dtype=torch.long)
    #     out = torch.gather(out, 1, label)
    #     loss = self.calculate_q_value_loss(out, reward)
    #     return loss

    def forward(self, state_dict):
        current_hand = state_dict['current_hand'].to(device)
        last_action = state_dict['last_action'].to(device)
        others_hand = state_dict['others_hand'].to(device)
        played_cards = state_dict['played_cards'].to(device)
        actions = state_dict['actions'].to(device)
        true_action = state_dict['true_action'].to(device)
        reward = state_dict['reward'].to(device)

        find_action_loss = self.find_action_task(current_hand, last_action, actions)
        predict_best_action_loss, calculate_q_value_loss = self.predict_best_action(current_hand, last_action,
                                                                                    played_cards, others_hand,
                                                                                    true_action, reward)
        total_loss = (self.n1 * find_action_loss + self.n2 * predict_best_action_loss + self.n3 * calculate_q_value_loss) / \
                     (self.n1 + self.n2 + self.n3)
        return total_loss

    def find_actions(self, current_hand, last_action):
        self.eval()
        with torch.no_grad():
            state_embeddings = self.encoder(current_hand) + self.encoder(last_action)
            out = self.linear2(torch.tanh(self.linear1(state_embeddings[:, 0, :])))
            return out









