import os
import string
import sys
import nltk
import torch
import dgl
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from detok_utils import detokenize
from table_utils import parse_table_to_text

nltk.download('stopwords')

bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class TableGNN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(TableGNN, self).__init__()
        self.conv1 = nn.Linear(in_feats, hidden_size)
        self.conv2 = nn.Linear(hidden_size, out_feats)
    
    def forward(self, features):
        x = torch.relu(self.conv1(features))
        x = torch.sigmoid(self.conv2(x))
        return x

def create_graph_from_data(data):
    g = dgl.graph(([], []))
    node_ids = {}
    node_idx = 0
    
    for k, v in data:
        if k not in node_ids:
            node_ids[k] = node_idx
            node_idx += 1
        if v not in node_ids:
            node_ids[v] = node_idx
            node_idx += 1
    
    g.add_nodes(len(node_ids))
    src_nodes = []
    dst_nodes = []
    for k, v in data:
        src_nodes.append(node_ids[k])
        dst_nodes.append(node_ids[v])
    
    g.add_edges(torch.tensor(src_nodes), torch.tensor(dst_nodes))
    g = dgl.add_self_loop(g)
    return g, len(node_ids)

def filter_with_gnn(data):
    g, num_nodes = create_graph_from_data(data)
    features = torch.eye(num_nodes)
    model = TableGNN(in_feats=num_nodes, hidden_size=16, out_feats=1)
    
    with torch.no_grad():
        predictions = model(features).squeeze()
    
    threshold = 0.3
    filtered_data = [data[i] for i in range(len(data)) if predictions[i] > threshold]
    return filtered_data

def convert_line_to_data_and_text(line):
    _, k, v, sent = line.strip().split("\t")
    k = k.replace("_$$_", " ").split("_||_")
    assert k[0] == "subj_title"
    k[0] = "title"
    assert k[1] == "subj_subtitle"
    k[1] = "subtitle"
    v = v.replace("_$$_", " ").split("_||_")
    data = [[kk, vv] for kk, vv in zip(k, v)]
    sent = sent.replace("_$$_", " ")
    return data, sent

def is_empty(x):
    return x.strip().replace(" ", "") == "n/a" or not any(t in string.ascii_letters + string.digits for t in x)

def is_counted_as_token(token):
    return (
        any(t in string.ascii_letters + string.digits + '.' for t in token)
        and not (all(t in string.ascii_letters for t in token) and len(token) == 1)
        and token not in stopwords.words('english')
    )

def is_value_hallucination(value, text):
    vtokens = [vv for vv in value.split() if is_counted_as_token(vv)]
    if not vtokens:
        vtokens = value.split()
    return any(vv not in text for vv in vtokens)

def rank_sentences_by_importance(text, data, top_n=5):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= top_n:
        return text  

    data_text = " ".join([f"{k}: {v}" for k, v in data])
    
    sent_embeddings = bert_model.encode(sentences, convert_to_tensor=True)
    data_embedding = bert_model.encode(data_text, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(data_embedding, sent_embeddings)[0].tolist()

    ranked_sentences = [s for _, s in sorted(zip(similarities, sentences), reverse=True)]
    return " ".join(ranked_sentences[:top_n])

def contrastive_filtering(data, text):
    assert data[0][0] == "title"
    title = data[0][1]
    assert data[1][0] == "subtitle"
    subtitle = data[1][1]

    data = {k: v for k, v in data[2:]}
    filtered_data = {k: v for k, v in data.items() if not is_empty(v)}
    filtered_data = {k: v for k, v in filtered_data.items() if not is_value_hallucination(v, text)}
    filtered_data = [("subtitle", subtitle)] + list(filtered_data.items())
    if not is_value_hallucination(title, text):
        filtered_data = [("title", title)] + filtered_data

    return filter_with_gnn(filtered_data)

if __name__ == "__main__":
    _, inp_suffix, oup_dir = sys.argv
    
    for inp_split, oup_split in [('train', 'train'), ('dev', 'valid'), ('test', 'test')]:
        with open(inp_suffix + '.' + inp_split) as f:
            lines = f.readlines()

        with open(os.path.join(oup_dir, f'{oup_split}.text'), 'w') as ftext, \
                open(os.path.join(oup_dir, f'{oup_split}.data'), 'w') as fdata:
            for line in lines:
                data, text = convert_line_to_data_and_text(line)
                print(data, text)
                text = rank_sentences_by_importance(text, data)
                ftext.write(detokenize(text.split(" ")).strip() + '\n')
                data = contrastive_filtering(data, text)
                print(data)
                data = [[detokenize(kk.split(" ")), detokenize(vv.split(" "))] for kk, vv in data]
                fdata.write(parse_table_to_text(data, one_line=True) + '\n')