import pickle
import matplotlib.pyplot as plt


with open('history.txt','rb') as file_pi:
    hist=pickle.load(file_pi)

plt.plot(hist['loss'],color='r')
plt.plot(hist['val_loss'],color='g')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'test_loss'], loc='upper left')
plt.savefig('loss.png')
plt.show()

plt.plot(hist['accuracy'],color='b')
plt.plot(hist['val_accuracy'],color='k')
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc'], loc='upper left')
plt.savefig('acc.png')
plt.show()
print("done")