from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


def bleu_score(reference, hypothesis):
    """
    Calculate BLEU score between a reference and hypothesis.

    Args:
        reference (str): The reference text.
        hypothesis (str): The generated text.

    Returns:
        float: BLEU score.
    """
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    return sentence_bleu([reference_tokens], hypothesis_tokens)


def rouge_scores(reference, hypothesis):
    """
    Calculate ROUGE scores between a reference and hypothesis.

    Args:
        reference (str): The reference text.
        hypothesis (str): The generated text.

    Returns:
        dict: ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)


def semantic_similarity(reference, hypothesis, model=None):
    """
    Calculate semantic similarity between reference and hypothesis using embeddings.

    Args:
        reference (str): The reference text.
        hypothesis (str): The generated text.
        model: Preloaded SentenceTransformer model for embeddings.

    Returns:
        float: Cosine similarity score.
    """
    if model is None:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    embeddings = model.encode([reference, hypothesis])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity


def score_response(reference, hypothesis, model=None):
    """
    Score a hypothesis against a reference using multiple metrics.

    Args:
        reference (str): The reference text.
        hypothesis (str): The generated text.
        model: Preloaded SentenceTransformer model for embeddings.

    Returns:
        dict: Scores for BLEU, ROUGE, and Semantic Similarity.
    """
    scores = {
        "BLEU": bleu_score(reference, hypothesis),
        "ROUGE": rouge_scores(reference, hypothesis),
        "SemanticSimilarity": semantic_similarity(reference, hypothesis, model),
    }
    return scores


if __name__ == "__main__":
    # Example usage
    reference = "The capital of France is Paris."
    hypothesis = "Paris is the capital of France."

    print("Loading semantic similarity model...")
    semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    print("Scoring response...")
    scores = score_response(reference, hypothesis, model=semantic_model)

    print("\nScores:")
    print(f"BLEU: {scores['BLEU']:.4f}")
    print(f"ROUGE-1: {scores['ROUGE']['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2: {scores['ROUGE']['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L: {scores['ROUGE']['rougeL'].fmeasure:.4f}")
    print(f"Semantic Similarity: {scores['SemanticSimilarity']:.4f}")
