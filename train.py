import torch
from torch import nn
import time
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from Model_CNN import CNN
from Dataset import DataSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
train_dataset_dir = './weather/train'
test_dataset_dir = './weather/test'

IMG_SIZE=224
batch_size = 8
learning_rate = 0.001
epoch = 30


traindataset = DataSet(train_dataset_dir,IMG_SIZE).dataset()
testdataset = DataSet(test_dataset_dir,IMG_SIZE).dataset()

num_of_classes = len(traindataset.classes)
print(traindataset.classes)

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False)

train_data_size=len(traindataset)
test_data_size = len(testdataset)

model_ft = CNN()
#model_ft=torchvision.models.resnet50()
model_ft = model_ft.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


writer = SummaryWriter("./logs")

#优化器
optimizer = torch.optim.Adam(model_ft.parameters(),
                            lr=learning_rate)

total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数

print(train_data_size)
start_time = time.time()
for i in range(epoch):
    start_time = time.time()
    # vgg16.train()
    print('------第{}轮训练开始---------'.format(i + 1))
    model_ft.train()
    #j=0
    for data in train_loader:
        imgs, targets = data
        #print(imgs.size())
        #writer.add_images('test',imgs,j)
        #j+=1
        #print(i)
        targets = targets.to(device)
        imgs = imgs.to(device)
        outputs = model_ft(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # vgg16.eval()
    model_ft.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model_ft(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1
    end_time = time.time()
    print(end_time - start_time)
    if (total_accuracy / test_data_size)>0.73 or (i+1)%10==0:  # 保存模型
        #best_acc = total_accuracy
        state={'model':model_ft,'modle_dict':model_ft.state_dict(),'optimizer':optimizer.state_dict(),'epoch':i+1}
        torch.save(state,"./models/cnn_epoch{}.pth".format(i + 1))  # 保存模型
writer.close()
