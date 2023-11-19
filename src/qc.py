# Use bert to encode the input
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader

from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import Birch

def get_embeddings(df, device="cuda", batch_size=32):
    embed_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    embed_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").half().eval().to(device)

    dataset = HFDataset.from_pandas(df)
    inputs = dataset.map(lambda example: embed_tokenizer(example["input"], padding="max_length", truncation=True, max_length=48, return_tensors="pt"), num_proc=32)
    inputs.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])

    embeddings = []
    dataloader = DataLoader(inputs, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    for batch in tqdm(dataloader):
        with torch.no_grad():
            outputs = embed_model(**batch)
            embeddings.append(outputs.last_hidden_state[:, 0, :].detach().cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    return embeddings

def reduce_dim(embeddings):
    tsne = TSNE(n_components=2, n_jobs=12, verbose=1)
    return tsne.fit_transform(embeddings)

def cluster_embeddings(embeddings, **kwargs):
    birch = Birch(**kwargs)
    birch.fit(embeddings)
    return birch.predict(embeddings), birch
