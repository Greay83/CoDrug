
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

def train(model_list,pred,loader,n_epoch,args):
    # 损失函数
    criterion = nn.MSELoss()

    # 预训模型开启冻结状态，线性层开启训练状态
    SMILES_model, seq_model, text_model, s2t_model, t2s_model = model_list
    SMILES_model.eval()
    seq_model.eval()
    text_model.eval()
    s2t_model.eval()
    t2s_model.eval()
    pred.train()

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
        seq = batch[2].cuda()
        aff = batch[3].cuda()

        # SMIELS嵌入
        with torch.no_grad():
            encode_input = {"encoder_input": idx.T, "encoder_pad_mask": mask.T}
            molecule_embedding = SMILES_model.encode(encode_input)
        molecule_embedding = latent_transform(molecule_embedding, mask, "cls")

        # seq嵌入
        with torch.no_grad():
            results = seq_model(seq, repr_layers=[33],
                            return_contacts=False)  # 因为这里esm1b有33层，所以输出33层的输出；不需要contacts回归，故false
        seq_embedding = results["representations"][33]
        seq_embedding = torch.mean(seq_embedding, dim=1)

        # seq嵌入后通过s2t
        seq_fusion_vector = s2t_model(seq_embedding)

        # Norm
        if args.norm:
            molecule_embedding = F.normalize(molecule_embedding, dim=-1)
            seq_fusion_vector = F.normalize(seq_fusion_vector, dim=-1)

        # dta预测
        pred = pred_model(molecule_embedding,seq_fusion_vector)

        optimizer.zero_grad()
        loss = criterion(pred.to(torch.float32), aff.to(torch.float32))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if args.verbose:
            train_step.set_postfix(loss=train_loss / (iter + 1))
        y = pred.cpu()
        label = aff.cpu()

        y = y.detach().numpy()
        label = label.detach().numpy()

        y_list.append(y)
        label_list.append(label)
    y = np.concatenate(y_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    ret = [rmse(y, label), mse(y, label), pearson(y, label)]
    print("TRAIN rmse:{}; mse:{}; pearson:{}".format(ret[0], ret[1], ret[2]))
    train_loss /= len(loader)
    print("loss=",train_loss)
    return train_loss,ret[0]
class MLP(nn.Module):
    def __init__(self, channel):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(channel, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, smi_latent, seq_latent):
        x = torch.cat((smi_latent, seq_latent), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x.squeeze()

class dta_dataset(Dataset):
    def __init__(self, fp = '../dataset/kiba_all.csv'):
        super(dta_dataset, self).__init__()
        # 从csv里读取三列数据
        df = pd.read_csv(fp)
        seq_raw_list = df["target_sequence"].tolist()
        SMILES_raw_list = df["compound_iso_smiles"].tolist()
        self.aff_list = df["affinity"].tolist()

        # SMLIES_tokenizer加载
        SMILES_tokenizer = MolEncTokeniser.from_vocab_file(r"../llm_ckpt/chemformer-ckpt/bart_vocab.txt", REGEX,
                                                           DEFAULT_CHEM_TOKEN_START)


        # 先判断是否已经存入。如果存在则读取，如果不存在则开始处理
        filepath, fullflname = os.path.split(fp)
        fname, ext = os.path.splitext(fullflname)
        save_fp = os.path.join(filepath, fname + "_processed.pkl")
        if os.path.exists(save_fp):
            with open(save_fp, 'rb') as f:
                data = pickle.load(f)
                self.SMILES_ids, self.SMILES_masks, self.seq_list = data[0],data[1],data[2]
        else:
            # smiles转化为idx和mas
            self.SMILES_ids, self.SMILES_masks = smiles_list_tokenizer(SMILES_raw_list, SMILES_tokenizer, 512)

            # seq转换成xx
            self.seq_list = [self.seq2token(seq,max_len=500) for seq in tqdm(seq_raw_list)]
            with open(save_fp, 'wb') as f:
                pickle.dump([self.SMILES_ids, self.SMILES_masks,self.seq_list],f)

    def __len__(self):
        return len(self.aff_list)

    def __getitem__(self, index):
        return (self.SMILES_ids[:,index],self.SMILES_masks[:,index],self.seq_list[index],self.aff_list[index])

    def seq2token(self,seq,max_len):
        if len(seq) > max_len:
            seq_item = torch.tensor(seq_tokenizer.encode(seq)[:500])
        elif len(seq) < max_len:
            pad = torch.zeros(max_len - len(seq)).int()
            seq_item = torch.cat((torch.tensor(seq_tokenizer.encode(seq)), pad))
        else:
            seq_item = torch.tensor(seq_tokenizer.encode(seq))
        return seq_item


    def collate_fn(self,batch):
        ids_list = [item[0] for item in batch]
        masks_list = [item[1] for item in batch]
        seq_list = [item[2] for item in batch]
        aff_list = [item[3] for item in batch]
        return (torch.stack(ids_list),torch.stack(masks_list),torch.stack(seq_list, dim=0),torch.tensor(np.array(aff_list)))


if __name__ == "__main__":
    #########################################参数设置##############################################
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
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


    # 预训练蛋白模型加载
    seq_model, seq_tokenizer = esm.pretrained.load_model_and_alphabet_local(
        "../llm_ckpt/esm/esm1b/esm1b_t33_650M_UR50S.pt")
    seq_model = seq_model.cuda(device)
    seq_dim = 1280

    # 预训练大语言模型加载
    if args.description_type == "pubmedbert-abs":
        text_tokenizer = AutoTokenizer.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract")
        text_model = AutoModel.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract").cuda(device)
        text_dim = 768
    elif args.description_type == "pubmedbert-full":
        text_tokenizer = AutoTokenizer.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        text_model = AutoModel.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext").cuda(device)
        text_dim = 768
    else:
        raise ValueError("no such description_model")

    # 线性层
    s2t_model = torch.nn.Linear(seq_dim, args.latent_dim).to(device)
    s2t_model.load_state_dict(torch.load("../model_save/step2_2024-04-26_14-13-31.336200/s2t.pt"))
    # molecule_to_description
    t2s_model = torch.nn.Linear(text_dim, args.latent_dim).to(device)
    t2s_model.load_state_dict(torch.load("../model_save/step2_2024-04-26_14-13-31.336200/t2s.pt"))

    # 预测器
    pred_model = MLP(molecule_dim + args.latent_dim).to(device)

    # 读取训练集、掩蔽
    dataset = dta_dataset()

    # 数据加载器
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, collate_fn=dataset.collate_fn)

    model_list = [SMILES_model, seq_model, text_model, s2t_model, t2s_model]

    # 参数冻结
    for model in model_list:
        for param in model.parameters():
            param.requires_grad = False

    model_param_group = [
        {"params": pred_model.parameters(), "lr": args.lr}]

    optimizer = torch.optim.Adam(model_param_group, weight_decay=args.decay)

    # 训练循环
    loss_list = []
    rmse_list = []

    for e in tqdm(range(1, args.epochs + 1)):
        loss,RMSE = train(model_list,pred_model,dataloader,e,args)
        loss_list.append(loss)
        rmse_list.append(RMSE)

    torch.save(pred_model.state_dict(), "../model_save/pred.pth")


