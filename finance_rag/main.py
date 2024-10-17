import warnings
warnings.filterwarnings("ignore")

import os
import sys
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
import faiss
from huggingface_hub import login
import pandas as pd 
from pathlib import Path
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from tqdm import tqdm
from eval_rag import evaluate_rag
import argparse

from utils import create_df_from_nodes, create_documents, load_data
from agents import RetrievalAgent

class Config:
   EMBED_DIMENSION =  1024
   EMBED_MODEL = "baconnier/Finance_embedding_large_en-V0.1"
   RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
   SIM_TOP_K = 50
   RERANKER_TOP_N = 30

cfg = Config()

# Llamaindex global settings for llm and embeddings
Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding(model_name=cfg.EMBED_MODEL)

DATA_MAP = {
    "financebench" : {
        "corpus" : "financebench_corpus.jsonl/corpus.jsonl",
        "queries" : "financebench_queries.jsonl/queries.jsonl",
        "gt" : "FinanceBench_qrels.tsv"
    },
    "finder" : {
        "corpus" : "finder_corpus.jsonl/corpus.jsonl",
        "queries" : "finder_queries.jsonl/queries.jsonl",
        "gt" : "FinDER_qrels.tsv"
    },
    "finqa" : {
        "corpus" : "finqa_corpus.jsonl/corpus.jsonl",
        "queries" : "finqa_queries.jsonl/queries.jsonl",
        "gt" : "FinQA_qrels.tsv"
    },
    "finqabench" : {
        "corpus" : "finqabench_corpus.jsonl/corpus.jsonl",
        "queries" : "finqabench_queries.jsonl/queries.jsonl",
        "gt" : "FinQABench_qrels.tsv"
    },
    "multiheirtt" : {
        "corpus" : "multiheirtt_corpus.jsonl/corpus.jsonl",
        "queries" : "multiheirtt_queries.jsonl/queries.jsonl",
        "gt" : "MultiHeirtt_qrels.tsv"
    },
    "tatqa" : {
        "corpus" : "tatqa_corpus.jsonl/corpus.jsonl",
        "queries" : "tatqa_queries.jsonl/queries.jsonl",
        "gt" : "TATQA_qrels.tsv"
    },
    "convfinqa" : {
        "corpus" : "convfinqa_corpus.jsonl/corpus.jsonl", 
        "queries" : "convfinqa_queries.jsonl/queries.jsonl",
        "gt" : "ConvFinQA_qrels.tsv"
    }

}

def evaluate_on_dataset(cfg, corpus, queries, gt, with_reranker=True):
    # Create FinQ Bench Documents
    documents = create_documents(corpus)

    # Initialize Retrieval Agent 
    ret_agent = RetrievalAgent(cfg=cfg, documents=documents)

    query_id_list = []
    corpus_id_list = []
    score_list = []

    for idx,row in queries.iterrows():
        query_id = row['_id']
        query_text = row['text']

        nodes = ret_agent.retrieve_nodes(query_text, with_reranker=with_reranker)
        # Extract top 10 unique corpus_id
        node_df = create_df_from_nodes(nodes)[:10]

        query_id_list.extend([query_id] * 10)
        corpus_id_list.extend(node_df.corpus_id.tolist())
        score_list.extend(node_df.score.tolist())


    final_df = pd.DataFrame({
        "query_id" : query_id_list, 
        "corpus_id" : corpus_id_list,
        "score" : score_list
    })

    # Convert the TSV data into a dictionary format for evaluation
    qrels_dict = gt.groupby('query_id').apply(lambda x: dict(zip(x['corpus_id'], x['score']))).to_dict()
    results = final_df.groupby('query_id').apply(lambda x: dict(zip(x['corpus_id'], x['score']))).to_dict()
    eval_results = evaluate_rag(qrels_dict, results, [1, 5, 10])
    for _ in eval_results:
        print(_)
    return final_df 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data-dir", type=Path, help="base data directory")
    parser.add_argument("--dataset-name", default=None, help="name of the dataset")
    parser.add_argument("--rerank", action='store_true', help="Flag to enable reranking")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory to save results")

    args = parser.parse_args()

    base_data_dir = args.base_data_dir
    assert os.path.exists(base_data_dir), f"Data Directory {base_data_dir} not found."

    dataset_name = args.dataset_name

    save_dir = args.save_dir
    if save_dir is None:
        raise ValueError("save directory can not be None. Provide valid --save-dir")
    
    save_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name is None: 
        print("Dataset name not provided. Inference on all datasets")
        for dataset_name, dataset_dict in DATA_MAP.items():
            print(f"Processing Dataset:{dataset_name} -- ReRank:{args.rerank}")
            corpus_path = base_data_dir / dataset_dict['corpus']
            queries_path = base_data_dir / dataset_dict['queries']
            gt_path = base_data_dir / dataset_dict['gt']
            assert os.path.exists(corpus_path), f"{corpus_path} does not exist"
            assert os.path.exists(queries_path), f"{queries_path} does not exist"
            assert os.path.exists(gt_path), f"{gt_path} does not exist"


            corpus, queries, gt = load_data(corpus_path=corpus_path,
                                            queries_path=queries_path,
                                            gt_path=gt_path)
            final_df = evaluate_on_dataset(cfg, corpus, queries, gt, with_reranker=args.rerank)
            file_name = dataset_name + '_results.csv'
            print(f"Saving result to {save_dir / file_name}")
            final_df.to_csv(save_dir / file_name, index=False)
            print("\n")

    else:
        dataset_dict = DATA_MAP.get(dataset_name, None)
        if dataset_dict is None:
            raise ValueError(f"{dataset_name} not found.")
        corpus_path = base_data_dir / dataset_dict['corpus']
        queries_path = base_data_dir / dataset_dict['queries']
        gt_path = base_data_dir / dataset_dict['gt']
        assert os.path.exists(corpus_path), f"{corpus_path} does not exist"
        assert os.path.exists(queries_path), f"{queries_path} does not exist"
        assert os.path.exists(gt_path), f"{gt_path} does not exist"




    