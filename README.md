# FinancialRAG
ACM-ICAIF '24 FinanceRAG Challenge

The Financial RAG Challenge aims to advance Retrieval-Augmented Generation (RAG) systems that can efficiently handle lengthy and complex financial documents. Participants are tasked with building systems capable of retrieving relevant contexts from large financial datasets and generating precise answers to financial queries, addressing real-world challenges such as financial terminology, industry-specific language, and numerical data.

# Installation

```bash
git clone https://github.com/ayush9818/FinancialRAG

conda create -n fin_env python=3.11
conda activate fin_env

cd FinancialRAG
pip install -r requirements.txt
```


# Submission

```bash
python collate_submission.py \
       --result-dir ../results/v1-base-rerank/ \
       --submission-path ../submission/v1-base-rerank.csv
```