# similarity_search

The code involves 

Text Splitting: The extracted text is split into smaller chunks for processing using RecursiveCharacterTextSplitter.

Text Embedding: The split text chunks are converted into numerical embeddings using HuggingFaceBgeEmbeddings.

Similarity Retrieval: The embeddings are used to perform similarity search, allowing users to input a query and retrieve relevant documents based on their similarity to the query.

Vector Store: The embeddings are stored in a vector store using Chroma, which facilitates efficient retrieval based on similarity.
