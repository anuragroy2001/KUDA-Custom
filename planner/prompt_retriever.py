import torch
import numpy as np
import clip

class PromptRetriever:
    def __init__(self):
        self.text_weight = 0.6
        self.top_k = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)

    def __call__(self, query, obs, examples):
        """
        examples: list of (query, obs, response)
        """
        queries = []
        images = []
        for example_query, example_image, _ in examples:
            queries.append(example_query)
            images.append(example_image)
        queries.append(query)
        images.append(obs)

        text_inputs = torch.cat([clip.tokenize(query) for query in queries]).to(self.device)
        image_inputs = torch.stack([self.preprocess(image) for image in images]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            image_features = self.model.encode_image(image_inputs)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_scores = (text_features[:-1] @ text_features[-1][..., None]).squeeze(-1)
        image_scores = (image_features[:-1] @ image_features[-1][..., None]).squeeze(-1)
        scores = image_scores + self.text_weight * text_scores

        top_k = scores.topk(self.top_k).indices.cpu().numpy()
        print("Matching prompts:", top_k)
        return [examples[index] for index in top_k]
