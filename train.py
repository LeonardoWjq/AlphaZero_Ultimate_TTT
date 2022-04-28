from neural_net import Network
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_split(ratio = 0.7, shuffle = False, seed = 0, is_regression = True):

    inners, outers, labels = torch.load('dataset_regression.pt') if is_regression else torch.load('dataset_classification.pt')

    assert len(inners) == len(outers) and len(outers) == len(labels)

    # shuffle the dataset 
    if shuffle:
        torch.manual_seed(seed)
        indices = torch.randperm(len(inners))
        inners = inners[indices]
        outers = outers[indices]
        labels = labels[indices]

    # point of split train/test
    split = int(len(inners)*ratio)
    train_inners, test_inners = inners[:split], inners[split:]
    train_outers, test_outers = outers[:split], outers[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    
    return train_inners, train_outers, train_labels, test_inners, test_outers, test_labels




def train(inners, outers, labels, network, criterion, optimizer, is_regression=True, epochs = 50, batch_size=32):
    network.to(device)
    inner_batches = torch.split(inners,batch_size)
    outer_batches = torch.split(outers,batch_size)
    labels_batches = torch.split(labels,batch_size)

    # regresion
    if is_regression:
        epoch_losses = []
        for epoch in tqdm(range(epochs)):
            total_loss = 0
            for inner, outer, label in zip(inner_batches, outer_batches, labels_batches):
                inner = inner.to(device)
                outer = outer.to(device)
                label = label.to(device)
                prediction = network(inner, outer)
                loss = criterion(prediction, label)
                # total batch_loss
                total_loss += loss.item()*len(inner)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            mean_loss = total_loss/len(inners)
            print(f'Epoch {epoch+1} MSE:', mean_loss)
            epoch_losses.append(mean_loss)

        torch.save(network.state_dict(),'regression_model.pt')

        with open('regression_losses.pickle','wb') as fp:
            pkl.dump(epoch_losses,fp)

    # classification
    else:
        epoch_accs = []
        for epoch in tqdm(range(epochs)):
            total_correct = 0
            for inner, outer, label in zip(inner_batches, outer_batches, labels_batches):
                inner = inner.to(device)
                outer = outer.to(device)
                label = label.to(device)

                # predicted probs
                pred_probs = network(inner, outer)
                # predicted classes by taking the max probable class
                pred_classes = torch.argmax(pred_probs,dim=1)

                # one-hot encoding of labels
                one_hot_labels = F.one_hot(label, num_classes=5).type(torch.float64)

                # cross entropy loss on predicted probabilities and one hot labels
                loss = criterion(pred_probs, one_hot_labels)

                # record how many are correct in this batch
                total_correct += torch.sum(pred_classes == label)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_accuracy = total_correct/len(inners)
            print(f'Epoch {epoch+1} Accuracy: {epoch_accuracy:.2%}')
            epoch_accs.append(epoch_accuracy)

        torch.save(network.state_dict(),'classification_model.pt')

        with open('classfication_accs.pickle','wb') as fp:
            pkl.dump(epoch_accs,fp)



def plot_figure(loss_curve = True):
    if loss_curve:
        with open('losses.pickle','rb') as fp:
            loss = pkl.load(fp)
            epochs = np.arange(1,len(loss)+1)
            plt.xticks(epochs)
            plt.plot(epochs,loss)
            plt.grid()
            plt.show()
            
def evalualte(test_inners, test_outers, test_labels, criterion, load = True):
    model = Network()
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
    with torch.no_grad():
        test_pred = model(test_inners, test_outers)
        test_loss = criterion(test_pred, test_labels)
        print('Test loss:', test_loss.item())
        print(test_pred[10:20].flatten())
        print(test_labels[10:20].flatten())
    
    return test_loss.item()
        



            



def main():
    train_inners, train_outers, train_labels, test_inners, test_outers, test_labels = load_and_split(ratio=0.01,shuffle=True,seed=1,is_regression=False)
    net = Network(is_regression=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=1e-3, weight_decay=1e-5)
    train(train_inners, train_outers, train_labels, net, criterion, optimizer, False, 100, 64)
    # evalualte(test_inners, test_outers, test_labels, criterion)
    # plot_figure()

if __name__ == '__main__':
    main()