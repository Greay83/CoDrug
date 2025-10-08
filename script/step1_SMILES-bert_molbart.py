import datetime
import json
import os.path
import time
from argparse import Namespace
import pandas as pd
import torch
import argparse
import warnings
import numpy as np
from torch import nn

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils.MolFormer_load import molformer_model
from transformers import BertTokenizer, BertModel, AutoTokenizer
from data.data_load import text2SMILES_Datasets
from utils.SciBert_load import prepare_text_tokens
from utils.mol_tokenizer import MolTranBertTokenizer
import torch.nn.functional as F
from molbart.tokeniser import MolEncTokeniser,DEFAULT_PAD_TOKEN
from molbart.models.pre_train import BARTModel
from molbart.decoder import DecodeSampler
from utils.utils import loss_fn, latent_transform, smiles_list_tokenizer

REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"
DEFAULT_CHEM_TOKEN_START = 272

def train(model_list,loader,n_epoch,args):
    # 预训模型开启冻结状态，线性层开启训练状态
    description_model, SMILES_model, m2d_model, d2m_model = model_list
    if args.llm_freeze:
        SMILES_model.eval()
        description_model.eval()
    else:
        SMILES_model.train()
        description_model.train()
    m2d_model.train()
    d2m_model.train()

    # 进度条设置
    if args.verbose:
        train_step = tqdm(loader,position=0)
        train_step.set_description('pretrain-epoch'+str(n_epoch))
    else:
        train_step = loader
    start_time = time.time()
    accum_loss = 0
    accum_acc = 0
    for iter,batch in enumerate(train_step):
        molecule_idx = batch[0].cuda()
        molecule_mask = batch[1].cuda()
        text_idx = batch[2].cuda()
        text_mask = batch[3].cuda()

        # 文本嵌入
        if args.llm_freeze:
            with torch.no_grad():
                description_output = description_model(input_ids=text_idx, attention_mask=text_mask)
        else:
            description_output = description_model(input_ids=text_idx, attention_mask=text_mask)
        description_repr = description_output["pooler_output"]

        description_fusion_vector = d2m_model(description_repr)

        # SMILES嵌入
        if args.llm_freeze:
            with torch.no_grad():
                encode_input = {"encoder_input": molecule_idx, "encoder_pad_mask": molecule_mask}
                molecule_embedding = SMILES_model.encode(encode_input)
        else:
            encode_input = {"encoder_input": molecule_idx, "encoder_pad_mask": molecule_mask}
            molecule_embedding = SMILES_model.encode(encode_input)

        molecule_embedding = latent_transform(molecule_embedding, molecule_mask,args.result_type)
        molecule_fusion_vector = m2d_model(molecule_embedding)

        # 后续计算
        loss, acc = loss_fn(description_fusion_vector, molecule_fusion_vector,args.loss_type)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss.item()
        accum_acc += acc
        torch.cuda.empty_cache()
        if args.verbose:
            train_step.set_postfix(loss=accum_loss/(iter+1), acc=accum_acc/(iter+1))

    accum_loss /= len(train_step)
    accum_acc /= len(train_step)
    print("CL Loss: {:.5f}\tCL Acc: {:.5f}\tTime: {:.5f}".format(accum_loss, accum_acc, time.time() - start_time))
    return accum_loss,accum_acc
def collate_function(batch):
    #tokenizer = MolTranBertTokenizer('bart_vocab.txt')
    smiles_list = [item[0] for item in batch]
    text_list = [item[1] for item in batch]

    # text
    description_tokenizer = BertTokenizer.from_pretrained('../llm_ckpt/scibert/allenai/scibert_scivocab_uncased/vocab.txt')
    text_tokens = prepare_text_tokens(description=text_list, tokenizer=description_tokenizer,
                                      max_seq_len=512)

    # smiles
    #smiles_tokens = tokenizer.batch_encode_plus(smiles_list, add_special_tokens=True,max_length=512,padding="max_length", truncation=True)
    #Tok = ChemformerTokenizer()
    #Tok.load_vocabulary(r"D:\pycharm files\GRmol_local\llm_ckpt\chemformer-ckpt\bart_vocab.txt")
    SMILES_tokenizer = MolEncTokeniser.from_vocab_file(r"..\llm_ckpt\chemformer-ckpt\bart_vocab.txt", REGEX, DEFAULT_CHEM_TOKEN_START)
    token_ids,SMILES_masks = smiles_list_tokenizer(smiles_list,SMILES_tokenizer,512)


    return (token_ids,SMILES_masks,text_tokens[0].clone().detach(), text_tokens[1].clone().detach())

class pub_dataset(Dataset):
    def __init__(self, CID2text_dir="../dataset/CID2text.json",
                 CID2SMILES_dir="../dataset/CID2SMILES.csv"):
        super(pub_dataset, self).__init__()
        self.CID2text_dir = CID2text_dir
        self.CID2SMILES_dir = CID2SMILES_dir
        self._load_data()
    def _load_data(self):
        # 读取cid2smiles文件
        df = pd.read_csv(self.CID2SMILES_dir)
        CID_list, self.SMILES_list = df["CID"].tolist(), df["SMILES"].tolist()

        # 这里是关于text加载
        with open("../dataset/CID2text.json", "r") as f:
            CID2text_data = json.load(f)

        # 读取cid2text文件
        self.text_list = []
        for index in tqdm(range(len(CID_list))):
            CID = CID_list[index]
            text = CID2text_data[str(CID)][0]
            self.text_list.append(text)

        # 确保两者数量相当
        assert len(self.text_list) == len(self.SMILES_list)

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        SMILES = self.SMILES_list[index]
        text = self.text_list[index]
        item = (SMILES, text)
        return item

if __name__ == "__main__":
    #########################################参数设置##############################################
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--description_lr", type=float, default=1e-4)
    parser.add_argument("--SMILES_lr", type=float, default=1e-4)
    parser.add_argument("--description_lr_scale", type=float, default=1)
    parser.add_argument("--SMILES_lr_scale", type=float, default=1)
    parser.add_argument("--device", type=str, default="cuda:0",choices=["cuda:0","cuda:1","cuda:2","cuda:3"])

    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--dataset_root", type=str, default='../dataset')
    parser.add_argument('--loss_type', type=str, default='EBM_NCE',choices=['EBM_NCE','InfoNCE',"cosine_sim_loss"])

    parser.add_argument("--description_type", type=str, default="scibert", choices=["scibert"])
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

    # 预训练大语言模型加载
    if args.description_type == "scibert":
        description_model = BertModel.from_pretrained('../llm_ckpt/scibert/allenai/scibert_scivocab_uncased').cuda(device)
        text_dim = 768
    else:
        raise ValueError("no such description_model")

    # 预训练SMILES模型加载
    if args.SMILES_type == "chemformer":
        SMILES_tokenizer = MolEncTokeniser.from_vocab_file(r"..\llm_ckpt\chemformer-ckpt\bart_vocab.txt", REGEX,
                                                           DEFAULT_CHEM_TOKEN_START)

        sampler = DecodeSampler(SMILES_tokenizer, 512)

        SMILES_model = BARTModel.load_from_checkpoint(checkpoint_path="../llm_ckpt/chemformer-ckpt/step=1000000.ckpt",
                                                      pad_token_idx=SMILES_tokenizer.vocab[SMILES_tokenizer.pad_token],
                                                      decode_sampler=sampler).cuda(device)
        molecule_dim = 512
    else:
        raise ValueError("no such SMILES_model")

    # 读取训练集、掩蔽
    dataset = pub_dataset(args.dataset_root)

    # 数据加载器
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, collate_fn=collate_function)
    # 线性转换层（这里后期可以改为MLP和transformer）
    if args.transform_type == "linear":
        # molecule_to_description
        m2d_model = torch.nn.Linear(molecule_dim, args.latent_dim).to(device)
        # molecule_to_description
        d2m_model = torch.nn.Linear(text_dim, args.latent_dim).to(device)
    elif args.transform_type == "MLP":
        pass
    else:
        raise ValueError("no such transform model")
    model_list = [description_model, SMILES_model, m2d_model, d2m_model]

    if args.llm_freeze:
        for param in description_model.parameters():
            param.requires_grad = False
        for param in SMILES_model.parameters():
            param.requires_grad = False
        model_param_group = [
            {"params": m2d_model.parameters(), "lr": args.description_lr * args.description_lr_scale},
            {"params": d2m_model.parameters(), "lr": args.SMILES_lr * args.SMILES_lr_scale},
        ]
    else:
        model_param_group = [
            {"params": description_model.parameters(), "lr": args.description_lr * args.description_lr_scale},
            {"params": SMILES_model.parameters(), "lr": args.SMILES_lr * args.SMILES_lr_scale},
            {"params": m2d_model.parameters(), "lr": args.description_lr * args.description_lr_scale},
            {"params": d2m_model.parameters(), "lr": args.SMILES_lr * args.SMILES_lr_scale},
        ]
    optimizer = torch.optim.Adam(model_param_group, weight_decay=args.decay)

    # 训练循环
    loss_list = []
    acc_list = []

    for e in tqdm(range(1, args.epochs + 1)):
        loss_e, acc_e=train(model_list,dataloader,e,args)
        loss_list.append(loss_e)
        acc_list.append(acc_e)


    save_dir = "../model_save/" + "step1_" + time_now + "/"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    torch.save(m2d_model.state_dict(), save_dir + 'm2d.pt')
    torch.save(d2m_model.state_dict(), save_dir + 'd2m.pt')

    log = pd.DataFrame(
        {'epoch': list(range(1, args.epochs + 1)), 'loss': loss_list, 'auc': acc_list})
    log.to_csv(save_dir + 'step1_data_record.csv', index=False)















