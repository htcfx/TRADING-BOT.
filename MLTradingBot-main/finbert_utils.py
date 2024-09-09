from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple, List

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news: List[str]) -> List[Tuple[torch.Tensor, str]]:
    results = []
    for text in news:
        if text:
            tokens = tokenizer(text, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
            result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
            probability = result[torch.argmax(result)]
            sentiment = labels[torch.argmax(result)]
            results.append((probability.item(), sentiment))
        else:
            results.append((0, labels[-1]))
    return results

if __name__ == "__main__":
    news_list = ['markets responded negatively to the news!', 'traders were displeased!']
    results = estimate_sentiment(news_list)
    for tensor, sentiment in results:
        print(f'Probability: {tensor}, Sentiment: {sentiment}')
    print("CUDA available:", torch.cuda.is_available())
