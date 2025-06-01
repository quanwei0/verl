```bash
cd verl
conda create -n search-r1 python=3.10 -y
conda activate search-r1

pip install -e .[sglang]
pip install flash-attn --no-build-isolation
```

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

```bash
conda activate retriever

save_path=/root/verl/data
python examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

```bash
conda activate retriever

save_path=/root/verl/data
index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu
```

```bash
conda activate search-r1
python examples/data_preprocess/preprocess_search_r1_dataset.py --local_dir ./data/searchR1_processed_direct
```