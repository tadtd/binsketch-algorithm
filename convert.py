import os
import gzip
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

def load_uci_to_dataframe(dataset_name, batch_size=1000, sample_size=None):
    """
    Load UCI bag-of-words format data into a pandas DataFrame with batch processing.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., 'bbc', 'enron', 'kos', 'nytimes')
    batch_size : int
        Number of documents to process at a time (default: 1000)
    sample_size : int, optional
        Number of documents to sample from the dataset (default: None, uses all documents)
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame where rows are documents and columns are words.
        Values are binary (0 or 1) indicating absence or presence of words.
    """
    data_path = f'./raw/{dataset_name}'
    vocab_file = f'{data_path}/vocab.{dataset_name}.txt'
    docword_file = f'{data_path}/docword.{dataset_name}.txt.gz'
    
    # Load vocabulary
    print(f"Loading vocabulary from {vocab_file}...")
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    
    # Load docword file - try gzipped first, then plain text
    print(f"Loading document-word data from {docword_file}...")
    try:
        # Try opening as gzipped file
        file_handle = gzip.open(docword_file, 'rt', encoding='latin-1')
        num_docs = int(file_handle.readline().strip())
        num_words = int(file_handle.readline().strip())
        num_nnz = int(file_handle.readline().strip())
    except (gzip.BadGzipFile, OSError):
        # If not gzipped, open as plain text
        print("  File is not gzipped, reading as plain text...")
        file_handle = open(docword_file, 'r', encoding='latin-1')
        num_docs = int(file_handle.readline().strip())
        num_words = int(file_handle.readline().strip())
        num_nnz = int(file_handle.readline().strip())
    
    print(f"  Documents: {num_docs}, Words: {num_words}, Non-zeros: {num_nnz}")
    
    # Apply sampling if requested
    if sample_size is not None and sample_size < num_docs:
        print(f"  Sampling {sample_size} documents out of {num_docs}")
        num_docs = sample_size
    
    # Use sparse matrix to save memory
    print("Creating sparse matrix...")
    sparse_matrix = lil_matrix((num_docs, num_words), dtype=np.int8)
    
    # Read and populate sparse matrix directly (memory efficient)
    for line in tqdm(file_handle, total=num_nnz, desc="Reading sparse data", unit=" entries"):
        parts = line.strip().split()
        if len(parts) == 3:
            doc_id, word_id, count = map(int, parts)
            # Only include documents within sample range
            if doc_id - 1 < num_docs:
                sparse_matrix[doc_id - 1, word_id - 1] = 1  # Binary: presence = 1
    
    file_handle.close()
    
    # Convert to dense format in batches to avoid memory issues
    print(f"Converting to DataFrame in batches of {batch_size} documents...")
    num_batches = (num_docs + batch_size - 1) // batch_size
    batch_dfs = []
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches", unit=" batch"):
        start_doc = batch_idx * batch_size
        end_doc = min((batch_idx + 1) * batch_size, num_docs)
        
        # Convert batch to dense and create DataFrame
        batch_matrix = sparse_matrix[start_doc:end_doc, :].toarray()
        batch_df = pd.DataFrame(batch_matrix, columns=vocab)
        batch_df.index = range(start_doc, end_doc)
        batch_dfs.append(batch_df)
    
    # Concatenate all batches
    print("Concatenating batches...")
    df = pd.concat(batch_dfs, axis=0)
    df.index.name = 'document_id'
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Sparsity: {(1 - df.sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
    
    return df

def process_folder_to_binary_uci(data_folder):
    print(f"Scanning folder: {data_folder}...")
    
    # --- 1. Read Raw Files ---
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

    # --- 2. Create Binary Bag of Words ---
    print("Vectorizing (Binary Mode)...")
    
    # binary=True forces all non-zero counts to be exactly 1
    vectorizer = CountVectorizer(stop_words='english', min_df=2, binary=True) 
    
    X = vectorizer.fit_transform(texts)
    vocab_list = vectorizer.get_feature_names_out()
    
    print(f"Vocabulary size: {len(vocab_list)} unique words.")

    # --- 3. Convert to UCI DataFrame ---
    print("Converting to UCI format...")
    
    X_coo = X.tocoo()
    
    df_uci = pd.DataFrame({
        'docID': X_coo.row + 1,
        'wordID': X_coo.col + 1,
        'count': X_coo.data  # These will now all be '1'
    })
    
    df_uci = df_uci.sort_values(by=['docID', 'wordID'])

    return df_uci, vocab_list, len(texts)

if __name__ == "__main__":
    # Example usage: Convert all datasets to DataFrames with batch processing
    datasets = ['bbc', 'enron', 'kos', 'nytimes']
    
    dataframes = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name.upper()} dataset")
        print(f"{'='*60}")
        
        try:
            # Adjust batch_size based on dataset size
            # Smaller batch for large datasets like nytimes
            batch_sizes = {
                'bbc': 500,
                'enron': 1000,
                'kos': 500,
                'nytimes': 5000
            }
            batch_size = batch_sizes.get(dataset_name, 1000)
            
            # Sample only 5000 documents for large datasets
            sample_sizes = {
                'nytimes': 5000,
                'enron': 5000
            }
            sample_size = sample_sizes.get(dataset_name, None)
            
            df = load_uci_to_dataframe(dataset_name, batch_size=batch_size, sample_size=sample_size)
            dataframes[dataset_name] = df
            
            print(f"\n{dataset_name.upper()} DataFrame Preview:")
            print(df.iloc[:5, :10])  # Show first 5 docs, first 10 words
            print(f"\nSample statistics:")
            print(f"  - Total documents: {len(df)}")
            print(f"  - Total unique words: {len(df.columns)}")
            print(f"  - Average words per document: {df.sum(axis=1).mean():.2f}")
            
            # Save to .npy format (more efficient than CSV)
            output_file = f'./data/{dataset_name}_binary.npy'
            vocab_file = f'./data/{dataset_name}_vocab.npy'
            
            print(f"Saving to {output_file}...")
            # Save the matrix data
            np.save(output_file, df.values)
            # Save the vocabulary
            np.save(vocab_file, df.columns.values)
            print(f"  Saved matrix to {output_file}")
            print(f"  Saved vocabulary to {vocab_file}")
            
        except FileNotFoundError as e:
            print(f"Skipping {dataset_name}: {e}")
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Loaded {len(dataframes)} datasets successfully!")
    print(f"Available DataFrames: {list(dataframes.keys())}")
    print(f"{'='*60}")