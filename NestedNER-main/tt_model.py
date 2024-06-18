import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class SpanModel(nn.Module):
    def __init__(self, bert_model_name, max_span_width, hidden_size, num_entities, entity_embedding_dim):
        super(SpanModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_span_width = max_span_width
        self.hidden_size = hidden_size

        self.span_embedding = nn.Linear(hidden_size * (max_span_width + 2), hidden_size)

        self.gcn_model = GCNModel(entity_embedding_dim, hidden_size, hidden_size)

        self.entity_embedding = nn.Embedding(num_entities, entity_embedding_dim)

    def forward(self, dataset, k):
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
                    span_tokens = tokens[i:(j+1)]  # Span tokens
                    span_token_ids = self.tokenizer.encode_plus(span_tokens, add_special_tokens=True, return_tensors='pt')['input_ids']
                    span_embedding = torch.max(embeddings[:, i:(j+1), :], dim=1)[0]  # Max pooling

                    span_input = torch.cat((span_embedding, cls_embedding.unsqueeze(1)), dim=1)
                    span_input = torch.cat((span_input, torch.tensor([[j-i+1]], dtype=torch.float)), dim=1)

                    span_inputs.append(span_input)

        token_inputs = torch.cat(token_inputs, dim=0)
        span_inputs = torch.cat(span_inputs, dim=0)

        # Apply span embedding layer
        span_representations = self.span_embedding(span_inputs)

        # Step 3: Build subgraphs
        subgraphs = build_subgraph(dataset, span_representations, k)

        # Step 4: Use GCN to update subgraph representations
        updated_subgraph_representations = update_subgraph_representation(self.gcn_model, subgraphs)

        # Step 5: Update span representations with entity embeddings
        updated_span_representations = update_span_representation(span_representations, updated_subgraph_representations)

        return token_inputs, updated_span_representations

    def get_top_k_similar_entities(self, span_representation, k):
        # Implement your logic to retrieve top-k similar entities for the given span representation
        pass

def build_subgraph(dataset, span_representations, k):
    subgraphs = []

    for i, span_representation in enumerate(span_representations):
        # Get top-k similar entities for the given span
        topk_entities = get_top_k_similar_entities(span_representation, k)

        # Create graph nodes for entities
        nodes = list(set([entity for entity, _ in topk_entities]))

        # Create edges between entities
        edges = []
        for entity, related_entities in topk_entities:
            for related_entity in related_entities:
                edges.append((entity, related_entity))
                edges.append((related_entity, entity))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(nodes, dtype=torch.float).view(-1, 1)

        # Create graph data object
        graph_data = Data(x=x, edge_index=edge_index)
        subgraphs.append(graph_data)

    return subgraphs

def update_subgraph_representation(model, subgraphs):
    updated_subgraph_representations = []

    for subgraph in subgraphs:
        x = subgraph.x
        edge_index = subgraph.edge_index

        updated_x = model(x, edge_index)
        updated_subgraph_representations.append(updated_x)

    return updated_subgraph_representations

def update_span_representation(span_representations, updated_subgraph_representations):
    updated_span_representations = []
    for i, span_representation in enumerate(span_representations):
        updated_span_representation = torch.cat([span_representation] + updated_subgraph_representations[i], dim=0)
        updated_span_representations.append(updated_span_representation)

    return updated_span_representations

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
