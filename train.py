from Network import Network
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl

def load_and_split(ratio=0.7, shuffle = False):
    inners, outers, labels = torch.load('dataset.pt')
    length = inners.size()[0]
    split = int(length*ratio)
    train_inners, test_inners = inners[:split], inners[split:]
    train_outers, test_outers = outers[:split], outers[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    
    return train_inners, train_outers, train_labels, test_inners, test_outers, test_labels





def train(inners, outers, labels, network, criterion, optimizer, epochs = 10):
    inner_batches = torch.split(inners,100)
    outer_batches = torch.split(outers,100)
    labels_batches = torch.split(labels,100)
    losses = []
    for epoch in tqdm(range(epochs)):
        all_pred = network(inners, outers)
        all_loss = criterion(all_pred, labels)
        losses.append(all_loss.item())
        print('Loss:',all_loss.item())
        for inner, outer, label in zip(inner_batches, outer_batches, labels_batches):
            prediction = network(inner, outer)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    torch.save(network.state_dict(),'model.pt')
    with open('losses.pickle','wb') as fp:
        pkl.dump(losses,fp)
        
        


def main():
    train_inners, train_outers, train_labels, test_inners, test_outers, test_labels = load_and_split(ratio=0.7)
    net = Network()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),0.001)
    train(train_inners, train_outers, train_labels,net,criterion,optimizer,20)
    test_pred = net(test_inners, test_outers)
    test_loss = criterion(test_pred,test_labels)
    print('Final loss:', test_loss.item())

if __name__ == '__main__':
    main()