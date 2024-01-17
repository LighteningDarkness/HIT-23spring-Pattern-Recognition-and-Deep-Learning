import matplotlib.pyplot as plt
def visualize(loss_list, val_loss_list, accuracy_list, name):
    plt.figure()
    plt.plot(loss_list, label='train_loss')
    plt.plot(val_loss_list, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(name+'loss.png')

    plt.figure()
    plt.plot(accuracy_list, label='accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(name+'accuracy.png')