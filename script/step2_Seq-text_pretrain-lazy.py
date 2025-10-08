import csv
import datetime
import json
import os.path
import time
from collections import defaultdict

import esm
import pandas as pd
import torch
import argparse
import warnings
import numpy as np

from torch import nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from utils.MolFormer_load import molformer_model
from transformers import BertTokenizer, BertModel
from data.data_load import text2SMILES_Datasets
from utils.SciBert_load import prepare_text_tokens
from utils.mol_tokenizer import MolTranBertTokenizer
import torch.nn.functional as F


def combine_attributes(attr, text_tokenizer, max_len=500):
    # num指的是batch大小，比如源码是3
    field = ["prot_name", "function", "subloc", "similarity"]
    sep_idx = text_tokenizer.sep_token_id
    cls_idx = text_tokenizer.cls_token_id
    pad_idx = text_tokenizer.pad_token_id
    # 先构建一个全为一的
    cls_ids = torch.ones(1, dtype=torch.long).unsqueeze(1) * cls_idx
    sep_ids = torch.ones(1, dtype=torch.long).unsqueeze(1) * sep_idx
    pad_ids = torch.ones(1, dtype=torch.long).unsqueeze(1) * pad_idx
    cls_ids = cls_ids.squeeze(1)
    sep_ids = sep_ids.squeeze(1)
    pad_ids = pad_ids.squeeze(1)

    # 这里一般version都为0，所以直接往下走；就算不为0，else就直接报错了
    # [CLS]是指开始，[sep]是指两个句子的隔断，[pad]是指填充
    # [CLS] attr1 [PAD] ... [PAD] [SEP] attr2 [PAD] ... [PAD] [SEP] attrn [PAD] ... [PAD]
    ids = [cls_ids]
    # 这不就是都叠加在一起的意思？？？？？？
    for k in field:
        ids.append(attr[k].long())
        ids.append(sep_ids)
    ids_len = int(torch.cat(ids[:-1], dim=-1).numel())
    if ids_len < max_len:
        pad_list = [pad_ids for i in range(max_len - ids_len)]
        ids.extend(pad_list)
        ids = torch.cat(ids[:-1], dim=-1)
    elif ids_len > max_len:
        ids = torch.cat(ids[:-1], dim=-1)
        ids = ids[:500]
    elif ids_len == max_len:
        ids = torch.cat(ids[:-1], dim=-1)

    masks_tep = ids != pad_idx
    masks = masks_tep.int()

    return ids, masks


def loss_fn(x1, x2, normalize=True, loss_type='cross_sim_loss', T=0.1):
    # 两个输入向量的条数应该一致
    assert x1.size()[0] == x2.size()[0]
    # 这里STM使用了正则化，向量除以自己的范数得到归一化向量
    if normalize:
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)

    if loss_type == 'cross_sim_loss':
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 查询有多少条数据（多少个分子）
        items = x1.size()[0]
        ############x为预测，y为模版#######################

        # 矩阵相乘：x(items,embedding)*y(items,embedding).Trans = z(items,items)
        logits = torch.mm(x1, x2.transpose(1, 0))  # B*B
        logits = torch.div(logits, T)
        labels = torch.arange(items).long().to(logits.device)  # B*1
        # 损失计算
        loss_1 = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        acc_1 = pred.eq(labels).sum().detach().cpu().item() * 1. / items

        ############y为预测，x为模版#######################
        # 矩阵相乘：x(items,embedding)*y(items,embedding).Trans = z(items,items)
        logits = torch.mm(x2, x1.transpose(1, 0))  # B*B
        logits = torch.div(logits, T)
        labels = torch.arange(items).long().to(logits.device)  # B*1
        # 损失计算
        loss_2 = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        acc_2 = pred.eq(labels).sum().detach().cpu().item() * 1. / items

        loss = (loss_1 + loss_2) / 2
        acc = (acc_1 + acc_2) / 2
    elif loss_type == 'reconstruct_loss':
        pass  # 这里后期再写，参照mol的pretrain
    elif loss_type == 'cosine_sim_loss':
        # 使用余弦相似度嵌入损失函数
        criterion = nn.CosineEmbeddingLoss()
        # 这里的输入是x1，x2和y
        # 由于这里的两个向量都是同一个实体，所以y取1即可，则loss=1-cosine(x1,x2)
        items = x1.size()[0]
        y = torch.ones(items)
        loss = criterion(x1, x2, y)
        acc = nn.functional.cosine_similarity(x1, x2).mean(dim=0)
        return loss, acc
    else:
        raise ValueError("no such loss function")
    return loss, acc


def train(model_list, loader, verbose):
    # 预训模型开启冻结状态，线性层开启训练状态
    seq_model, text_model, s2t_model, t2s_model = model_list
    if llm_freeze:
        seq_model.eval()
        text_model.eval()
    else:
        seq_model.train()
        text_model.train()
    s2t_model.train()
    t2s_model.train()

    # 进度条设置
    if verbose:
        train_step = tqdm(loader)
    else:
        train_step = loader
    start_time = time.time()
    accum_loss = 0
    accum_acc = 0
    for batch in train_step:
        text_tokens = batch[0].cuda()
        text_masks = batch[1].cuda()
        seq_tokens = batch[2].cuda()

        # 文本嵌入
        # 这里是否需要torch.no_grad?
        if llm_freeze:
            with torch.no_grad():
                description_output = text_model(input_ids=text_tokens, attention_mask=text_masks)
        else:
            description_output = text_model(input_ids=text_tokens, attention_mask=text_masks)

        description_repr = description_output["pooler_output"]
        description_fusion_vector = t2s_model(description_repr)

        # SEQ嵌入
        if llm_freeze:
            with torch.no_grad():
                results = seq_model(seq_tokens, repr_layers=[33],
                                    return_contacts=False)  # 因为这里esm1b有33层，所以输出33层的输出；不需要contacts回归，故false
        else:
            results = seq_model(seq_tokens, repr_layers=[33],
                                return_contacts=False)  # 因为这里esm1b有33层，所以输出33层的输出；不需要contacts回归，故false
        seq_embedding = results["representations"][33]
        seq_embedding = torch.mean(seq_embedding, dim=1)

        seq_fusion_vector = s2t_model(seq_embedding)
        # 后续计算
        loss, acc = loss_fn(description_fusion_vector, seq_fusion_vector)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss.item()
        accum_acc += acc

    accum_loss /= len(train_step)
    accum_acc /= len(train_step)
    print("CL Loss: {:.5f}\tCL Acc: {:.5f}\tTime: {:.5f}".format(accum_loss, accum_acc, time.time() - start_time))
    return accum_loss, accum_acc


class seq_dataset(Dataset):
    def __init__(self, tsv_file, text_tokenizer, verbose, device):
        super(seq_dataset, self).__init__()
        self.tsv_file = tsv_file
        self.text_tokenizer = text_tokenizer
        self.verbose = verbose
        self.device = device
        self._load_data()

    def _load_data(self):
        # 读取cid2smiles文件
        with open(self.tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            # 下面的数据处理不能离开with语句的缩进，不然就关闭了
            # 这里是读取下一次迭代（如果还没有进行迭代，那应该是读取第0条），这里应该是返回列表的表题
            fields = next(reader)
            # tqdm，这里最后一个参数是读取行数，不然tqdm没有进度条
            reader = iter(tqdm(reader, desc="data loading", total=self.get_line_count(self.tsv_file)))

            # 这里创建一个空列表和空字典，分别用于存储序列和文本
            sequences = []
            texts = defaultdict(list)

            seq_field = "Sequence"
            text_fields = ["ProteinName", "Function", "SubcellularLocation", "Similarity"]
            self.text_field2acronym = {"ProteinName": "prot_name", "Function": "function",
                                       "SubcellularLocation": "subloc", "Similarity": "similarity"}

            # 计数
            seq = 0
            text = 0

            # 紧接着开启一个循环，i是次序，values是tsv中的一条数据
            for i, values in enumerate(reader):
                # 把前面提取的fileds以及values里的前len(fields)列数据打包为一个字典
                for field, value in zip(fields, values[:len(fields)]):

                    # 如果field=seq_field，也就是说如果这一条是序列，则直接把value存进sequences中
                    if field == seq_field:
                        sequences.append(value)
                        seq += 1
                    # 再次判断，
                    elif field in text_fields:
                        text += 1
                        texts[self.text_field2acronym[field]].append(value)
            assert seq == text / 4, "序列和文本数量不对等"
        # 这里是return数量
        self.seq = seq

        # 至此已经全部读取
        # 下面是对list和dict遍历，进行处理
        text_ids_items = []
        text_masks_items = []
        sequences_items = []
        if self.verbose:
            process = tqdm(range(self.seq), desc="data encoding")
        else:
            process = range(self.seq)
            print("data encoding")
        for i in process:
            text_item = {v: torch.tensor(self.text_tokenizer.encode(texts[v][i],
                                                                    max_length=128,
                                                                    truncation=True, add_special_tokens=False))
                         for v in self.text_field2acronym.values()}

            ids, masks = combine_attributes(text_item, self.text_tokenizer, max_len=500)
            seq_item = self.seq2token(sequences[i], max_len=500)

            text_ids_items.append(ids)
            text_masks_items.append(masks)
            sequences_items.append(seq_item)

        del sequences
        del texts
        self.text_ids_items = text_ids_items
        self.text_masks_items = text_masks_items
        self.sequences_items = sequences_items

    def __len__(self):
        return self.seq

    def __getitem__(self, index):
        return (self.text_ids_items[index], self.text_masks_items[index], self.sequences_items[index])

    def seq2token(self, seq, max_len):
        if len(seq) > max_len:
            seq_item = torch.tensor(seq_tokenizer.encode(seq)[:500])
        elif len(seq) < max_len:
            pad = torch.zeros(max_len - len(seq)).int()
            seq_item = torch.cat((torch.tensor(seq_tokenizer.encode(seq)), pad))
        else:
            seq_item = torch.tensor(seq_tokenizer.encode(seq))
        return seq_item

    def collate_fn(self, batch):
        ids_list = [item[0] for item in batch]
        masks_list = [item[1] for item in batch]
        seq_list = [item[2] for item in batch]

        return (torch.stack(ids_list, dim=0), torch.stack(masks_list, dim=0), torch.stack(seq_list, dim=0))

    def get_line_count(self, file_name, chunk_size=8192 * 1024):
        """
        Get the number of lines in a file.

        Parameters:
            file_name (str): file name
            chunk_size (int, optional): chunk size for reading large files
        """
        count = 0
        with open(file_name, "rb") as fin:
            chunk = fin.read(chunk_size)
            while chunk:
                count += chunk.count(b"\n")
                chunk = fin.read(chunk_size)
        return count


if __name__ == "__main__":
    #########################################参数设置##############################################
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--text_lr", type=float, default=0.00001)
    parser.add_argument("--seq_lr", type=float, default=0.00001)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

    parser.add_argument("--verbose", type=bool, default=False)

    parser.add_argument("--dataset_root", type=str, default='../dataset')
    parser.add_argument("--tsv_file", type=str, default=r'../dataset/uniprot_sprot_filtered.tsv')

    parser.add_argument("--description_type", type=str, default="pubmedbert-abs",
                        choices=["pubmedbert-abs", "pubmedbert-full"])
    parser.add_argument("--seq_type", type=str, default="esm-1b", choices=["esm-1b"])
    parser.add_argument("--transform_type", type=str, default="linear", choices=["linear", "MLP", "transform"])
    parser.add_argument("--llm_freeze", type=bool, default=True, choices=[True, False])

    # 各个向量的维度
    parser.add_argument("--latent_dim", type=int, default=768)

    args = parser.parse_args()
    print("arguments\t", args)

    seed = args.seed
    device_num = args.device
    batch_size = args.batch_size
    epochs = args.epochs
    decay = args.decay

    text_lr = args.text_lr
    seq_lr = args.seq_lr

    description_type = args.description_type
    seq_type = args.seq_type
    transform_type = args.transform_type
    llm_freeze = args.llm_freeze

    latent_dim = args.latent_dim
    verbose = args.verbose
    tsv_file = args.tsv_file
    #########################################警告屏蔽#############################################
    warnings.filterwarnings("ignore")  # 老是弹出什么CSR警告，一点卵用都没有，直接屏蔽掉
    # 时间
    time_now = str(datetime.datetime.now())
    time_now = time_now.replace(" ", "_")
    time_now = time_now.replace(":", "-")

    # 设置cuda，设置随机种子
    device = torch.device(device_num)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    # 预训练大语言模型加载
    if description_type == "pubmedbert-abs":
        text_tokenizer = AutoTokenizer.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract")
        text_model = AutoModel.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract").cuda(device)
        text_dim = 768
    elif description_type == "pubmedbert-full":
        text_tokenizer = AutoTokenizer.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        text_model = AutoModel.from_pretrained(
            "../llm_ckpt/pubmedbert/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext").cuda(device)
        text_dim = 768

    else:
        raise ValueError("no such description_model")

    # esm蛋白预训练模型加载
    if seq_type == "esm-1b":
        seq_model, seq_tokenizer = esm.pretrained.load_model_and_alphabet_local(
            "../llm_ckpt/esm/esm1b/esm1b_t33_650M_UR50S.pt")
        seq_model = seq_model.cuda(device)
        seq_dim = 1280
    else:
        raise ValueError("no such seq_model")

    # 读取训练集、掩蔽
    dataset = seq_dataset(tsv_file, text_tokenizer, verbose, device=device)

    # 数据加载器
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=8, prefetch_factor=2,collate_fn=dataset.collate_fn)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, collate_fn=dataset.collate_fn)
    if transform_type == "linear":
        # molecule_to_description
        s2t_model = torch.nn.Linear(seq_dim, latent_dim).to(device)
        # molecule_to_description
        t2s_model = torch.nn.Linear(text_dim, latent_dim).to(device)
    elif transform_type == "MLP":
        pass
    elif transform_type == "transformer":
        pass
    else:
        raise ValueError("no such transform model")

    model_list = [seq_model, text_model, s2t_model, t2s_model]

    if llm_freeze:
        model_param_group = [
            {"params": s2t_model.parameters(), "lr": seq_lr},
            {"params": t2s_model.parameters(), "lr": text_lr},
        ]
    else:
        model_param_group = [
            {"params": seq_model.parameters(), "lr": seq_lr},
            {"params": text_model.parameters(), "lr": text_lr},
            {"params": s2t_model.parameters(), "lr": seq_lr},
            {"params": t2s_model.parameters(), "lr": text_lr},
        ]

    # 优化器设置
    optimizer = torch.optim.Adam(model_param_group, weight_decay=decay)

    # 训练循环
    loss_list = []
    acc_list = []

    for e in tqdm(range(1, epochs + 1)):
        loss_e, acc_e = train(model_list, dataloader, verbose)
        loss_list.append(loss_e)
        acc_list.append(acc_e)

    save_dir = "../model_save/" + "step2_" + time_now + "/"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(s2t_model.state_dict(), save_dir + 's2t.pt')
    torch.save(t2s_model.state_dict(), save_dir + 't2s.pt')

    log = pd.DataFrame(
        {'epoch': list(range(1, epochs + 1)), 'loss': loss_list, 'auc': acc_list})
    log.to_csv(save_dir + 'step2_data_record.csv', index=False)
