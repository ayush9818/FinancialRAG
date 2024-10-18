import pandas as pd 
import os 
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, help="Directory which contains results csv files")
    parser.add_argument("--submission-path", type=Path, default=None, help="Path to save the submission file")

    args = parser.parse_args()

    result_dir = args.result_dir 
    submission_path = args.submission_path 
    
    assert os.path.exists(result_dir), f"{result_dir} does not exist. Please provide valid result-dir"
    
    print(f"Result Files Directory:{result_dir.resolve()}\nSubmission File Path:{submission_path.resolve()}\n")
    
    if submission_path is None:
        raise ValueError(f"Provide valid --submission-path")
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    submission_df = pd.DataFrame(columns=['query_id','corpus_id'])
    for file_name in os.listdir(result_dir):
        file_path = result_dir / file_name
        df = pd.read_csv(file_path)[['query_id','corpus_id']]
        submission_df = pd.concat([submission_df, df])

    submission_df.to_csv(submission_path, index=False)

    
