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

def load_and_split(train_ratio=0.8, val_ratio = 0.1, shuffle = False, seed = 0, is_regression = True):

    inners, outers, labels = torch.load('dataset_regression_proof.pt') if is_regression else torch.load('dataset_classification_proof.pt')

    assert len(inners) == len(outers) and len(outers) == len(labels)

    # shuffle the dataset 
    if shuffle:
        torch.manual_seed(seed)
        indices = torch.randperm(len(inners))
        inners = inners[indices]
        outers = outers[indices]
        labels = labels[indices]

    # point of split train/test
    train_size = int(len(inners)*train_ratio)
    val_size = int(len(inners)*val_ratio)

    train_inners = inners[:train_size]
    train_outers = outers[:train_size]
    train_labels = labels[:train_size]

    val_inners = inners[train_size:train_size+val_size]
    val_outers = outers[train_size:train_size+val_size]
    val_labels = labels[train_size:train_size+val_size]

    test_inners = inners[train_size+val_size:]
    test_outers = outers[train_size+val_size:]
    test_labels = labels[train_size+val_size:]
    
    return train_inners, train_outers, train_labels, val_inners, val_outers, val_labels, test_inners, test_outers, test_labels




def train(inners_train,outers_train,labels_train,inners_val,outers_val,labels_val,network,criterion,optimizer,is_regression=True,epochs = 50,batch_size=32):
    network.to(device)

    inner_batches = torch.split(inners_train,batch_size)
    outer_batches = torch.split(outers_train,batch_size)
    labels_batches = torch.split(labels_train,batch_size)

    inners_val = inners_val.to(device)
    outers_val = outers_val.to(device)
    labels_val = labels_val.to(device)

    # regresion
    if is_regression:
        train_losses = []
        val_losses = []
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
            
           
            # get validation loss
            val_pred = network(inners_val, outers_val)
            val_mse = criterion(val_pred, labels_val).item()

            train_mse = total_loss/len(inners_train)

            print(f'epoch {epoch+1} training MSE:', train_mse)
            print(f'epoch {epoch+1} validation MSE:', val_mse)

            train_losses.append(train_mse)
            val_losses.append(val_mse)

        torch.save(network.state_dict(),'regression_model.pt')

        with open('regression_losses.pickle','wb') as fp:
            pkl.dump((train_losses, val_losses),fp)
    # classification
    else:
        train_accs = []
        val_accs = []
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
                one_hot_labels = F.one_hot(label, num_classes=3).type(torch.float64)

                # cross entropy loss on predicted probabilities and one hot labels
                loss = criterion(pred_probs, one_hot_labels)

                # record how many are correct in this batch
                total_correct += torch.sum(pred_classes == label)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            train_accuracy = total_correct/len(inners_train)

            val_probs = network(inners_val, outers_val)
            val_pred = torch.argmax(val_probs, dim=1)
            val_accuracy = torch.sum(val_pred == labels_val).item()/len(labels_val)

            print(f'epoch {epoch+1} training accuracy: {train_accuracy:.2%}')
            print(f'epoch {epoch+1} validation accuracy: {val_accuracy:.2%}')

            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)
            

        torch.save(network.state_dict(),'classification_model.pt')

        with open('classification_accs.pickle','wb') as fp:
            pkl.dump((train_accs, val_accs),fp)



def plot_figure(is_regression=True):
    if is_regression:
        with open('regression_losses.pickle','rb') as fp:
            train_loss, val_loss = pkl.load(fp)
            epochs = np.arange(1,len(train_loss)+1)
            plt.xticks(np.arange(0,len(train_loss)+1,2))
            plt.plot(epochs,train_loss,label='training')
            plt.plot(epochs,val_loss,label='validation')
            plt.title('Learning Curve of the Regression Model')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error')
            plt.legend()
            plt.grid()
            plt.savefig("regression_loss.png")
    else:
        with open('classification_accs.pickle','rb') as fp:
            train_accs, val_accs = pkl.load(fp)
            epochs = np.arange(1,len(train_accs)+1)
            plt.xticks(np.arange(0,len(train_accs)+1,2))
            plt.plot(epochs,train_accs,label='training')
            plt.plot(epochs,val_accs,label='validation')
            plt.title('Learning Curve of the Classification Model')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid()
            plt.savefig("classification_acc.png")
            
def evalualte(test_inners, test_outers, test_labels, criterion, is_regression=True):
    model = Network(is_regression)
    if is_regression:
        model.load_state_dict(torch.load('regression_model.pt'))
        model.eval()
        with torch.no_grad():
            test_pred = model(test_inners, test_outers)
            test_loss = criterion(test_pred, test_labels)
            mse = test_loss.item()
            print('Test MSE:', mse)
        return mse
    else:
        model.load_state_dict(torch.load('classification_model.pt'))
        model.eval()
        with torch.no_grad():
            pred_prob = model(test_inners, test_outers)
            pred_labels = torch.argmax(pred_prob, dim=1)
            acc = torch.sum(pred_labels == test_labels).item()/len(test_labels)
            print(f"Test Accuracy: {acc:.2%}")
        return acc 
        



            



def main():
    regression = False
    train_inners,train_outers,train_labels,val_inners,val_outers,val_labels,test_inners,test_outers,test_labels = load_and_split(train_ratio=0.9,val_ratio=0.01, shuffle=True,seed=1,is_regression=regression)
    net = Network(is_regression=regression)
    criterion = nn.MSELoss() if regression else nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=2e-4, weight_decay=1e-5)
    train(train_inners,train_outers,train_labels,val_inners,val_outers,val_labels,net,criterion,optimizer,is_regression=regression,epochs=30,batch_size=64)
    evalualte(test_inners, test_outers, test_labels, criterion, is_regression=regression)
    plot_figure(is_regression=regression)
    

if __name__ == '__main__':
    main()