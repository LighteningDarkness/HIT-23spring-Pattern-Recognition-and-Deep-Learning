import torch
import os
import numpy as np
from torch.autograd import Variable
def train(train_loader, val_loader, optimizer, criterion, model, epoch, save,mixup=False):

    accuracy_list = []
    loss_list = []
    val_loss_list = []
    for e in range(epoch):
        #train
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
            if mixup:
                images_copy=Variable(images.clone(),requires_grad=True)
                labels_copy=Variable(labels.clone())
                #alpha=0.5
                lam = np.random.beta(0.5, 0.5)
                index = torch.randperm(inputs.shape[0]).cuda()
                images_copy=lam*images+(1-lam)*images_copy[index,:]
                labels_copy=lam*labels+(1-lam)*labels_copy[index,:]
                output_mixup=model(images_copy)
                loss_mixup=criterion(output_mixup,labels_copy)
                loss_mixup.backward(retain_graph=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        loss_list.append(loss)
        #val
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data

                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)
                val_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss_list.append(val_loss)
        accuracy = 100.0 * correct / total
        #保存最好的模型
        if save:
            if e == 0:
                best_accuracy = accuracy
                torch.save(model.state_dict(), save)
            else:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), save)

        #打印loss和accuracy
        print('epoch: %d, loss:%.3f, val_loss: %.3f, accuracy: %.3f' % (e + 1, loss, val_loss, accuracy))
        # 记录每个epoch的准确率
        accuracy_list.append(accuracy)

    return accuracy_list, loss_list, val_loss_list
        
def test(test_loader, model):
    result = []
    with torch.no_grad():
        for data in test_loader:
            images, name = data
            images = images.cuda()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            result.extend([name, predicted])
    
    return result