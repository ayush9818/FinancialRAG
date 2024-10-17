import pandas as pd 
from llama_index.core import Document

def create_documents(df):
    """Create Documents with metadata from df"""
    documents = []
    for idx,row in df.iterrows():
        _ = Document(
            text=row['text'], 
            metadata={'_id' : row['_id'], 'title' : row['title']}
            )
        documents.append(_)
    return documents

def load_data(corpus_path, queries_path, gt_path):
    corpus = pd.read_json(corpus_path, lines=True)
    queries = pd.read_json(queries_path, lines=True)
    gt_path = pd.read_csv(gt_path, sep='\t')

    return corpus, queries, gt_path


def create_df_from_nodes(nodes, extract_unique=True):
    init_rows = []
    for node in nodes:
        tmp = {
            "score" : node.score,
            "text" : node.text,
            "corpus_id" : node.metadata['_id']
        }
        init_rows.append(tmp)
    tmp_df = pd.DataFrame(init_rows)

    if not extract_unique:
        return tmp_df 
    
    final_rows = []
    for corpus_id, corpus_df in tmp_df.groupby('corpus_id'):
        max_score = corpus_df['score'].max()
        text = corpus_df[corpus_df.score == max_score].text.tolist()[0]
        final_rows.append({
            "corpus_id" : corpus_id, 
            "text" : text, 
            "score" : max_score
        })
    df = pd.DataFrame(final_rows)
    df = df.sort_values(by='score', ascending=False)
    return df


if __name__ == "__main__":
    pass