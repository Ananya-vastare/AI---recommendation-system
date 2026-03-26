import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')
nltk.download('punkt')
allstopwords = stopwords.words('english')

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def get_dummy_paper_data(search_query: str):
    """
    Process a search query, generate embeddings, and return dummy paper data.
    """
    tokenize_text = word_tokenize(search_query)
    filtered_text = [word for word in tokenize_text if word.isalpha() and word not in allstopwords]
    processed_query = " ".join(filtered_text)
    query_embeddings = model.encode(processed_query)
    dummy_paper = {
        "title": "A Dummy Paper on AI Research",
        "authors": ["Alice Smith", "Bob Johnson"],
        "summary": "This paper explores dummy approaches to artificial intelligence, focusing on placeholder data for testing pipelines.",
        "published": "2026-03-25",
        "arxiv_id": "dummy-1234"
    }

    summary_tokens = word_tokenize(dummy_paper['summary'])
    filtered_summary = [word for word in summary_tokens if word.isalpha() and word not in allstopwords]
    processed_summary = " ".join(filtered_summary)

    summary_embeddings = model.encode(processed_summary)

    return {
        "query_embeddings": query_embeddings,
        "paper_details": dummy_paper,
        "summary_embeddings_shape": summary_embeddings.shape
    }
