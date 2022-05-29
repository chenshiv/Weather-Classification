import os
import torch
from torchvision import transforms
from PIL import Image
i=0 #识别图片计数
root_path="testimgs"         #待测试文件夹
names=os.listdir(root_path)
IMG_SIZE=224
for name in names:
    print(name)
    i=i+1

    data_class=['dew', 'fogsmog', 'frost', 'lightning', 'rain', 'rainbow', 'sandstorm', 'snow']
    #按文件索引顺序排列
    image_path=os.path.join(root_path,name)
    image=Image.open(image_path)
    image = image.convert("RGB")
    #print(image)
    transforms_data = transforms.Compose([  # 数据处理
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std = [0.5, 0.5, 0.5]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    image=transforms_data(image)
    #print(image.shape)


    #model=CNN()
    checkpoint=torch.load("./models/CNN_epoch25.pth",map_location=torch.device("cpu")) #选择训练后得到的模型文件

    model=checkpoint['model']
    # print(model)
    image=torch.reshape(image,(1,3,IMG_SIZE,IMG_SIZE))      #修改待预测图片尺寸，需要与训练时一致
    model.eval()
    with torch.no_grad():
        output=model(image)
    #print(output)
    #print(data_class)#输出预测结果
    #print(int(output.argmax(1)))
    print("第{}张图片预测为：{}".format(i,data_class[int(output.argmax(1))]))

