"""
Dataset Conversion Utility
Converts UCI bag-of-words datasets to binary matrix format.
"""

import os
import gzip
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer


class UCIDatasetLoader:
    """Loader for UCI bag-of-words format datasets."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def _load_vocabulary(self, dataset_name: str) -> List[str]:
        """Load vocabulary from text file."""
        vocab_file = self.data_path / dataset_name / f'vocab.{dataset_name}.txt'
        print(f"Loading vocabulary from {vocab_file}...")
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    
    def _open_docword_file(self, dataset_name: str):
        """Open document-word file, handling both gzipped and plain text."""
        docword_file = self.data_path / dataset_name / f'docword.{dataset_name}.txt.gz'
        print(f"Loading document-word data from {docword_file}...")
        
        try:
            file_handle = gzip.open(docword_file, 'rt', encoding='latin-1')
            file_handle.readline()  # Test read
            file_handle.seek(0)
            return file_handle
        except (gzip.BadGzipFile, OSError):
            print("  File is not gzipped, reading as plain text...")
            return open(docword_file, 'r', encoding='latin-1')
    
    def _read_metadata(self, file_handle) -> Tuple[int, int, int]:
        """Read dataset metadata from file header."""
        num_docs = int(file_handle.readline().strip())
        num_words = int(file_handle.readline().strip())
        num_nnz = int(file_handle.readline().strip())
        print(f"  Documents: {num_docs}, Words: {num_words}, Non-zeros: {num_nnz}")
        return num_docs, num_words, num_nnz
    
    def _populate_sparse_matrix(self, file_handle, sparse_matrix, num_docs: int, num_nnz: int):
        """Populate sparse matrix from document-word data."""
        for line in tqdm(file_handle, total=num_nnz, desc="Reading sparse data", unit=" entries"):
            parts = line.strip().split()
            if len(parts) == 3:
                doc_id, word_id, _ = map(int, parts)
                if doc_id - 1 < num_docs:
                    sparse_matrix[doc_id - 1, word_id - 1] = 1  # Binary representation
    
    def _convert_to_dataframe(self, sparse_matrix, vocab: List[str], batch_size: int) -> pd.DataFrame:
        """Convert sparse matrix to DataFrame in batches."""
        num_docs = sparse_matrix.shape[0]
        num_batches = (num_docs + batch_size - 1) // batch_size
        batch_dfs = []
        
        print(f"Converting to DataFrame in batches of {batch_size} documents...")
        for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit=" batch"):
            start_doc = batch_idx * batch_size
            end_doc = min((batch_idx + 1) * batch_size, num_docs)
            
            batch_matrix = sparse_matrix[start_doc:end_doc, :].toarray()
            batch_df = pd.DataFrame(batch_matrix, columns=vocab)
            batch_df.index = range(start_doc, end_doc)
            batch_dfs.append(batch_df)
        
        print("Concatenating batches...")
        df = pd.concat(batch_dfs, axis=0)
        df.index.name = 'document_id'
        return df
    
    def load_dataset(self, 
                    dataset_name: str, 
                    batch_size: int = 1000, 
                    sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load UCI bag-of-words dataset into binary DataFrame.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'bbc', 'enron', 'kos', 'nytimes')
            batch_size: Number of documents to process at a time
            sample_size: Optional number of documents to sample
        
        Returns:
            DataFrame with binary word presence (rows=docs, columns=words)
        """
        vocab = self._load_vocabulary(dataset_name)
        file_handle = self._open_docword_file(dataset_name)
        
        num_docs, num_words, num_nnz = self._read_metadata(file_handle)
        
        if sample_size is not None and sample_size < num_docs:
            print(f"  Sampling {sample_size} documents out of {num_docs}")
            num_docs = sample_size
        
        print("Creating sparse matrix...")
        sparse_matrix = lil_matrix((num_docs, num_words), dtype=np.int8)
        
        self._populate_sparse_matrix(file_handle, sparse_matrix, num_docs, num_nnz)
        file_handle.close()
        
        df = self._convert_to_dataframe(sparse_matrix, vocab, batch_size)
        
        print(f"DataFrame shape: {df.shape}")
        sparsity = (1 - df.sum().sum() / (df.shape[0] * df.shape[1])) * 100
        print(f"Sparsity: {sparsity:.2f}%")
        
        return df


def process_folder_to_binary_uci(data_folder: str) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """
    Process a folder of text files into binary UCI format.
    
    Args:
        data_folder: Path to folder containing text files
    
    Returns:
        Tuple of (UCI DataFrame, vocabulary array, document count)
    """
    print(f"Scanning folder: {data_folder}...")
    
    texts = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='latin-1') as f:
                        texts.append(f.read())
                except Exception as e:
                    print(f"Skipped {file}: {e}")

    print(f"Read {len(texts)} documents.")
    
    print("Vectorizing (Binary Mode)...")
    vectorizer = CountVectorizer(stop_words='english', min_df=2, binary=True)
    X = vectorizer.fit_transform(texts)
    vocab_list = vectorizer.get_feature_names_out()
    
    print(f"Vocabulary size: {len(vocab_list)} unique words.")
    
    print("Converting to UCI format...")
    X_coo = X.tocoo()
    df_uci = pd.DataFrame({
        'docID': X_coo.row + 1,
        'wordID': X_coo.col + 1,
        'count': X_coo.data
    })
    df_uci = df_uci.sort_values(by=['docID', 'wordID'])

    return df_uci, vocab_list, len(texts)

def save_dataset(df: pd.DataFrame, dataset_name: str, output_path: str = './data'):
    """Save dataset to numpy format."""
    output_dir = Path(output_path)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    matrix_file = output_dir / f'{dataset_name}_binary.npy'
    vocab_file = output_dir / f'{dataset_name}_vocab.npy'
    
    print(f"Saving to {matrix_file}...")
    np.save(matrix_file, df.values)
    np.save(vocab_file, df.columns.values)
    print(f"  Saved matrix to {matrix_file}")
    print(f"  Saved vocabulary to {vocab_file}")


def print_dataset_summary(df: pd.DataFrame, dataset_name: str):
    """Print summary statistics for a dataset."""
    print(f"\n{dataset_name.upper()} DataFrame Preview:")
    print(df.iloc[:5, :10])
    print(f"\nStatistics:")
    print(f"  - Total documents: {len(df)}")
    print(f"  - Total unique words: {len(df.columns)}")
    print(f"  - Average words per document: {df.sum(axis=1).mean():.2f}")


def main():
    """Main execution function."""
    RAW_DATA_PATH = './raw'
    OUTPUT_DATA_PATH = './data'
    
    DATASETS = ['bbc', 'enron', 'kos', 'nytimes']
    
    BATCH_SIZES = {
        'bbc': 500,
        'enron': 1000,
        'kos': 500,
        'nytimes': 5000
    }
    
    SAMPLE_SIZES = {
        'nytimes': 5000,
        'enron': 5000
    }
    
    loader = UCIDatasetLoader(RAW_DATA_PATH)
    dataframes = {}
    
    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()} dataset")
        print(f"{'='*60}")
        
        try:
            batch_size = BATCH_SIZES.get(dataset_name, 1000)
            sample_size = SAMPLE_SIZES.get(dataset_name, None)
            
            df = loader.load_dataset(
                dataset_name=dataset_name,
                batch_size=batch_size,
                sample_size=sample_size
            )
            
            dataframes[dataset_name] = df
            print_dataset_summary(df, dataset_name)
            save_dataset(df, dataset_name, OUTPUT_DATA_PATH)
            
        except FileNotFoundError as e:
            print(f"Skipping {dataset_name}: {e}")
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Successfully processed {len(dataframes)}/{len(DATASETS)} datasets")
    print(f"Available DataFrames: {list(dataframes.keys())}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()