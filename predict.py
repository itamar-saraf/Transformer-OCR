import torch
from torch.autograd import Variable
import numpy as np
from model import make_model
from dataset import char2token, token2char
from dataset import subsequent_mask
import cv2
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

test_path = {'SVT': 'eval/SVT-Perspective',
             'CUTE80': 'eval/CUTE80',
             'IC15': 'eval/IC15/images'}

test_gt = {'SVT': 'eval/SVT-Perspective/gt.txt',
           'CUTE80': 'eval/CUTE80/gt.txt',
           'IC15': 'eval/IC15/gt.txt'}

model_path = {'0m': 'checkpoint/m00000000_0.240851.pth',
              '1m': 'checkpoint/m00000001_0.189284.pth',
              '2m': 'checkpoint/m00000002_0.168973.pth',
              '3m': 'checkpoint/m00000003_0.153565.pth',
              '0f': 'checkpoint/00000000_0.224805.pth',
              '1f': 'checkpoint/00000001_0.180550.pth'}

model = make_model(len(char2token))
model.load_state_dict(torch.load(model_path['3m'], map_location=torch.device('cpu')))
model.to(device)
model.eval()
src_mask = Variable(torch.from_numpy(np.ones([1, 1, 36], dtype=np.bool)).to(device))
SIZE = 96


def greedy_decode(src, max_len=36, start_symbol=1):
    global model
    global src_mask
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).long().to(device)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .long().to(device)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).long().to(device).fill_(next_word)], dim=1)
        if token2char[next_word.item()] == '>':
            break
    ret = ys.cpu().numpy()[0]
    out = [token2char[i] for i in ret]
    out = "".join(out[1:-1])
    return out


def resize(img):
    h, w, c = img.shape
    if w > h:
        nw, nh = SIZE, int(h * SIZE / w)
        if nh < 10: nh = 10
        # print(h, w, nh, nw)
        img = cv2.resize(img, (nw, nh))
        a1 = int((SIZE - nh) / 2)
        a2 = SIZE - nh - a1
        pad1 = np.zeros((a1, SIZE, c), dtype=np.uint8)
        pad2 = np.zeros((a2, SIZE, c), dtype=np.uint8)
        img = np.concatenate((pad1, img, pad2), axis=0)
    else:
        nw, nh = int(w * SIZE / h), SIZE
        if nw < 10: nw = 10
        # print(h, w, nh, nw)
        img = cv2.resize(img, (nw, nh))
        a1 = int((SIZE - nw) / 2)
        a2 = SIZE - nw - a1
        pad1 = np.zeros((SIZE, a1, c), dtype=np.uint8)
        pad2 = np.zeros((SIZE, a2, c), dtype=np.uint8)
        img = np.concatenate((pad1, img, pad2), axis=1)
    return img


def do_folder(root):
    hit = 0
    all = 0
    for line in open(root).readlines():
        all += 1
        imp, label = line.strip('\n').split('\t')
        imp = os.path.join(test_path['IC15'], imp)
        img = cv2.imread(imp)
        img = resize(img) / 255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)
        pred = greedy_decode(img)
        if pred != label:
            hit += 1
            print('imp:', imp, 'label:', label, 'pred:', pred, hit, all, hit / all)
        else:
            print('imp:', imp, 'label:', label, 'pred:', pred)
    print(hit, all, hit / all)


if __name__ == '__main__':
    do_folder(test_gt['IC15'])
