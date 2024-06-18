import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class SpanModel(nn.Module):
    def __init__(self, bert_model_name, max_span_width, hidden_size):
        super(SpanModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_span_width = max_span_width
        self.hidden_size = hidden_size

        self.span_embedding = nn.Linear(hidden_size * (max_span_width + 2), hidden_size)

    def forward(self, dataset):
        token_inputs = []
        span_inputs = []

        for example in dataset:
            tokens = example['tokens']
            entities = example['entities']

            # Step 1: Tokenize input and get BERT embeddings
            tokenized_inputs = self.tokenizer.encode_plus(tokens, add_special_tokens=True, return_tensors='pt')
            input_ids = tokenized_inputs['input_ids']
            attention_mask = tokenized_inputs['attention_mask']

            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
            embeddings = outputs[0]  # BERT embeddings
            cls_embedding = embeddings[:, 0, :]  # [CLS] embedding

            token_inputs.append(cls_embedding)

            # Step 2: Generate span representations
            for i in range(len(tokens)):
                for j in range(i, min(i + self.max_span_width, len(tokens))):
                    span_tokens = tokens[i:(j + 1)]  # Span tokens
                    span_token_ids = \
                    self.tokenizer.encode_plus(span_tokens, add_special_tokens=True, return_tensors='pt')['input_ids']
                    span_embedding = torch.max(embeddings[:, i:(j + 1), :], dim=1)[0]  # Max pooling

                    span_input = torch.cat((span_embedding, cls_embedding.unsqueeze(1)), dim=1)
                    span_input = torch.cat((span_input, torch.tensor([[j - i + 1]], dtype=torch.float)), dim=1)

                    span_inputs.append(span_input)

        token_inputs = torch.cat(token_inputs, dim=0)
        span_inputs = torch.cat(span_inputs, dim=0)

        # Apply span embedding layer
        span_representations = self.span_embedding(span_inputs)

        return token_inputs, span_representations


# Example usage
dataset = [...]  # Your dataset
model = SpanModel('bert-base-uncased', 5, 768)
token_embeddings, span_representations = model(dataset)
