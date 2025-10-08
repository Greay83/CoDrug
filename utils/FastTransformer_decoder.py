from fast_transformers.builders.transformer_builders import TransformerDecoderBuilder
import torch
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class molecule_decoder(torch.nn.Module):
    def __init__(self):
        super(molecule_decoder, self).__init__()
        decoder_layer = TransformerDecoderLayer(d_model=768,
                                                nhead=12,
                                                dim_feedforward=768,
                                                dropout=0.1,
                                                activation="relu", )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)

        '''
        builder = TransformerDecoderBuilder.from_kwargs(n_layers=6,  # 层数
                                                        n_heads=12,  # 头数
                                                        query_dimensions=64,  # query的维度：d_emd维度/头数
                                                        value_dimensions=64,  # value的维度：d_emd维度/头数
                                                        feed_forward_dimensions=768,  # 前向层，维度也是d_emd维度
                                                        dropout=0.1,
                                                        attention_dropout=0.1,
                                                        self_attention_type="linear",  # 使用线性作为注意力模型，还可以是full，到时候查查区别
                                                        activation="gelu")
        self.blocks = builder.get()
        '''

        # 这里的输入维度之后要修改成len(vocab.txt)
        self.tgt_emb = torch.nn.Embedding(2362, 768)
        self.linear = torch.nn.Linear(768, 2362)

    def forward(self, tgt, memory, mask):
        tgt_embeddings = self.tgt_emb(tgt)
        mask = mask.T
        test = mask.bool()
        output = self.transformer_decoder(tgt_embeddings, memory,tgt_key_padding_mask=~test,memory_key_padding_mask=~test)
        output = self.linear(output)
        return output
