import matplotlib.pyplot as plt

def visualize_loss_in_epoch(train_loss, val_loss):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.grid()
    plt.legend(labels=['train_loss', "val_loss"])
    plt.show()
    plt.savefig("plots/train_val_loss_in_epoch.png")