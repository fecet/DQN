import torch
import torch.optim as optim

from models.simple_dqn import calculate_q_value_dqn
from utils import gen_data_loader
from tqdm import tqdm

device = torch.device('cuda:0')
model = calculate_q_value_dqn().to(device)
optimizer = optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 2e-5},
    {'params': list(set(model.parameters()) - set(model.encoder.parameters())),
     'lr': 1e-4}
])


def train(train_model, epochs, data_dir):
    train_loader = gen_data_loader(data_dir)
    for epoch in range(epochs):
        iters = 0
        for state_dict in tqdm(train_loader):
            train_model.train()
            optimizer.zero_grad()
            loss = train_model(state_dict)
            loss.backward()
            optimizer.step()
            iters += 1
            if iters % 200 == 0:
                print('epochs: ', epoch, ' loss:', loss.item())
    torch.save(model.state_dict(), 'saved_model.pkl')
