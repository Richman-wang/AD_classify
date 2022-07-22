import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
# from model import convnext_tiny as create_model

from model import convnext_base as create_model

os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

def main(model,filepath):
    file_write = open('../data_v4/predict_res_0711_cov.txt','w')

    img_size = 224
    # data_transform = transforms.Compose(
    #     [transforms.Resize(img_size),
    #      transforms.CenterCrop(img_size),
    #      transforms.ToTensor(),
    #      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transform = transforms.Compose(
        [transforms.CenterCrop((img_size,img_size)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    filelist = os.listdir(filepath)
    for img in tqdm(filelist):
        img_path = os.path.join(filepath, img) # 图片路径
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path).convert("RGB")
        # plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            score = torch.softmax(output, dim=0)[0].cpu().item()
            predict_cla = torch.argmax(predict).numpy()
        img_nme = str(os.path.basename(img_path).split('.')[0]) # 图片的id
        file_write.write(str(img_nme)+'\t'+str(score)+'\t'+str(class_indict[str(predict_cla)])+ '\n')
    file_write.close()

        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                             predict[predict_cla].numpy())
        # # plt.title(print_res)
        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                             predict[i].numpy()))
        # plt.show()

if __name__ == '__main__':
    # create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "./weights_0711/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    filepath = '../data_v4/test_all'
    main(model,filepath)