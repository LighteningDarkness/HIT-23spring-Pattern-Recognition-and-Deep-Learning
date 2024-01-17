import torch
from transformers import BertTokenizer, BertModel
from torch import nn
class BERT4SA(nn.Module):

    # 使用bert-base-cased，编码器具有12个隐层，输出768维张量，12个自注意力头，共110M参数量，在不区分大小写的英文文本上进行训练而得到。
    def __init__(self, output_dim, pretrained_name='bert-base-cased'):

        super(BERT4SA, self).__init__()
        
        # 使用预训练BERT模型
        self.bert = BertModel.from_pretrained(pretrained_name)

        # 线性维度变换
        self.ln = nn.Linear(768, output_dim)
        #softmax映射到[0,1]区间
        self.sm=nn.Softmax(dim=1)

    def forward(self, tokens_X):

        # 得到最后一层的 '<cls>' 信息， 其标志全部上下文信息
        res = self.bert(**tokens_X)
        res_linear=self.ln(res[1]) 
        #print(res[0].shape)
        r=self.sm(res_linear)
        #print(r)
        return r
