
import datetime
import os.path
import pickle
import time
import esm
import argparse
import warnings
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
from molbart.tokeniser import MolEncTokeniser,DEFAULT_PAD_TOKEN
from molbart.models.pre_train import BARTModel
from molbart.decoder import DecodeSampler
from transformers import AutoTokenizer, AutoModel

from utils.utils import loss_fn, latent_transform, smiles_list_tokenizer
import torch.nn.functional as F

REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
DEFAULT_CHEM_TOKEN_START = 272
def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def train(model_list,loader,n_epoch,args):
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 预训模型开启冻结状态，线性层开启训练状态
    SMILES_model, pred_model = model_list
    SMILES_model.train()
    pred_model.train()

    # 进度条设置
    if args.verbose:
        train_step = tqdm(loader, position=0)
        train_step.set_description('pretrain-epoch' + str(n_epoch))
    else:
        train_step = loader

    # 训练步
    y_list = []
    label_list = []
    train_loss = 0
    for iter,batch in enumerate(train_step):
        idx = batch[0].cuda()
        mask = batch[1].cuda()
        label = batch[2].cuda()

        # SMIELS嵌入
        with torch.no_grad():
            encode_input = {"encoder_input": idx.T, "encoder_pad_mask": mask.T}
            molecule_embedding = SMILES_model.encode(encode_input)
        molecule_embedding = latent_transform(molecule_embedding, mask, "cls")

        # 预测
        pred = pred_model(molecule_embedding)

        optimizer.zero_grad()
        loss = criterion(pred.to(torch.float32), label.to(torch.float32))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss
def eval(model_list,loader):
    # 预训模型开启冻结状态，线性层开启训练状态
    SMILES_model, pred_model = model_list
    SMILES_model.eval()
    pred_model.eval()

    # 设置2个空列表
    y_true = []  # 样本的真实标签，形状（样本数，）
    y_prob = []  # 预测为1的概率值，形状（样本数，）,即结果预测为1的概率
    y_scores = []  # 预测标签（归一后，非0即1）
    # for循环每一个batch，无梯度计算y_pred(在这里就是y_scores)
    for iter,batch in enumerate(loader):  # 从总数据集中取出各个batch
        idx = batch[0].cuda()
        mask = batch[1].cuda()
        label = batch[2].cuda()

        # SMIELS嵌入
        with torch.no_grad():
            encode_input = {"encoder_input": idx.T, "encoder_pad_mask": mask.T}
            molecule_embedding = SMILES_model.encode(encode_input)
        molecule_embedding = latent_transform(molecule_embedding, mask, "cls")

        # 无梯度传播
        with torch.no_grad():
            pred = pred_model(molecule_embedding)
        #label = torch.reshape(label, (-1, 2))


        y_true.append(label[:, 1])
        y_prob.append(pred[:, 1])
        #y_scores.append(pred[:, 1])
        # 将所有tensor并在一起

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_prob = torch.cat(y_prob, dim=0).cpu().numpy()
    #y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    # 将y_scores四舍五入（因为之前已经进行了softmax,所以大于0.5就为1）
    # y_scores = np.around(y_scores, 0).astype(int)  # .around()是四舍五入的函数 第二个参数0表示保留0位小数，也就只保留整数

    return roc_auc_score(y_true, y_prob)


class pred_MLP(nn.Module):
    def __init__(self, channel):
        super(pred_MLP, self).__init__()
        self.fc1 = nn.Linear(channel, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, smi_latent):
        x = self.fc1(smi_latent)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x.squeeze()

class mol_dataset(Dataset):
    def __init__(self, csv = r'D:\pycharm files\GRmol_local\dataset\HIV.csv'):
        super(mol_dataset, self).__init__()
        # 从csv里读取三列数据
        df = pd.read_csv(csv)
        self.SMILES_list = df["smiles"].tolist()
        label_raw_list = df["HIV_active"].tolist()

        # SMLIES_tokenizer加载
        SMILES_tokenizer = MolEncTokeniser.from_vocab_file(r"../llm_ckpt/chemformer-ckpt/bart_vocab.txt", REGEX,
                                                           DEFAULT_CHEM_TOKEN_START)
        # 先判断是否已经存入。如果存在则读取，如果不存在则开始处理
        filepath, fullflname = os.path.split(csv)
        fname, ext = os.path.splitext(fullflname)
        save_fp = os.path.join(filepath, fname + "_processed.pkl")
        if os.path.exists(save_fp):
            with open(save_fp, 'rb') as f:
                data = pickle.load(f)
                self.SMILES_ids, self.SMILES_masks, self.label_list = data[0],data[1],data[2]
        else:
            # smiles转化为idx和mas
            self.SMILES_ids, self.SMILES_masks = smiles_list_tokenizer(self.SMILES_list, SMILES_tokenizer, 512)
            self.label_list = []
            for y in label_raw_list:
                label = torch.nn.functional.one_hot(torch.tensor(y).to(torch.int64), num_classes=2)  # 这里要转成int64不然要报错
                self.label_list.append(label)
            with open(save_fp, 'wb') as f:
                pickle.dump([self.SMILES_ids, self.SMILES_masks,self.label_list],f)

    def __len__(self):
        return len(self.SMILES_list)

    def __getitem__(self, index):
        return (self.SMILES_ids[:,index],self.SMILES_masks[:,index],self.label_list[index])

    def collate_fn(self,batch):
        ids_list = [item[0] for item in batch]
        masks_list = [item[1] for item in batch]
        y_list = [item[2] for item in batch]
        return (torch.stack(ids_list),torch.stack(masks_list),torch.tensor(np.array(y_list)))


if __name__ == "__main__":
    #########################################参数设置##############################################
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda:0",choices=["cuda:0","cuda:1","cuda:2","cuda:3"])
    parser.add_argument("--norm", type=bool, default=True)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--dataset_root", type=str, default='../dataset')
    parser.add_argument('--loss_type', type=str, default='EBM_NCE',choices=['EBM_NCE','InfoNCE',"cosine_sim_loss"])

    parser.add_argument("--description_type", type=str, default="pubmedbert-abs", choices=["pubmedbert-abs"])
    parser.add_argument("--SMILES_type", type=str, default="chemformer", choices=["chemformer"])
    parser.add_argument("--transform_type", type=str, default="linear", choices=["linear", "MLP", "transform"])
    parser.add_argument("--result_type", type=str, default="pooler", choices=["pooler", "cls"])

    parser.add_argument("--llm_freeze", type=bool, default=True, choices=[True,False])

    # 各个向量的维度
    parser.add_argument("--latent_dim", type=int, default=768)

    args = parser.parse_args()
    print("arguments\t", args)
    warnings.filterwarnings("ignore")  # 老是弹出什么CSR警告，一点卵用都没有，直接屏蔽掉

    # 时间
    time_now = str(datetime.datetime.now())
    time_now = time_now.replace(" ", "_")
    time_now = time_now.replace(":", "-")

    # 设置cuda，设置随机种子
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)


    # 预训练SMILES模型加载
    SMILES_tokenizer = MolEncTokeniser.from_vocab_file(r"../llm_ckpt/chemformer-ckpt/bart_vocab.txt", REGEX,
                                                       DEFAULT_CHEM_TOKEN_START)

    sampler = DecodeSampler(SMILES_tokenizer, 512)

    SMILES_model = BARTModel.load_from_checkpoint(checkpoint_path="../llm_ckpt/chemformer-ckpt/step=1000000.ckpt",
                                                  pad_token_idx=SMILES_tokenizer.vocab[SMILES_tokenizer.pad_token],
                                                  decode_sampler=sampler).cuda(device)
    molecule_dim = 512


    # 线性层
    '''
    s2t_model = torch.nn.Linear(seq_dim, args.latent_dim).to(device)
    s2t_model.load_state_dict(torch.load("../model_save/step2_2024-04-26_14-13-31.336200/s2t.pt"))
    '''

    # 预测器
    pred_model = pred_MLP(molecule_dim).to(device)


    # 读取训练集、掩蔽
    dataset = mol_dataset()

    # 数据加载器
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, collate_fn=dataset.collate_fn)

    model_list = [SMILES_model, pred_model]

    '''
    # 参数冻结
    for param in SMILES_model.parameters():
        param.requires_grad = False
    '''

    model_param_group = [
        {"params": pred_model.parameters(), "lr": args.lr}]

    optimizer = torch.optim.Adam(model_param_group, weight_decay=args.decay)

    # 训练循环
    loss_list = []
    rmse_list = []

    for e in tqdm(range(1, args.epochs + 1)):
        train_loss = train(model_list,dataloader,e,args)
        train_auc = eval(model_list,dataloader)
        print(train_loss,train_auc)

    #torch.save(pred_model.state_dict(), "../model_save/pred.pth")


