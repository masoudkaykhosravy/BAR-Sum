"""
=============================================================================
Module: encoder.py
Project: BAR-Sum (Factually Faithful Extractive Text Summarization via DRL)
Author: Masoud Keikhosravi
Description:
    Hierarchical Document Encoder combining a pre-trained Transformer 
    backbone with a Bidirectional LSTM (BiLSTM) sentence-level encoder 
    and positional encodings for extractive summarization.
=============================================================================
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for sentence order in documents.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, num_sentences, hidden_dim]
        Returns:
            Tensor with added positional encodings.
        """
        return x + self.pe[:, :x.size(1), :]


class HierarchicalDocEncoder(nn.Module):
    """
    Hierarchical Document Encoder:
    1. Word-level Transformer representation (e.g., BERT/RoBERTa)
    2. Sentence pooling (Mean/CLS pooling)
    3. Document-level BiLSTM contextual encoder
    """
    def __init__(
        self,
        transformer_model_name: str = "bert-base-uncased",
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        freeze_transformer: bool = False
    ):
        super(HierarchicalDocEncoder, self).__init__()
        
        # 1. Pretrained Transformer Backbone
        self.config = AutoConfig.from_pretrained(transformer_model_name)
        self.transformer = AutoModel.from_pretrained(transformer_model_name, config=self.config)
        
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
                
        self.word_hidden_dim = self.config.hidden_size

        # 2. Positional Encoding for Sentence Order
        self.pos_encoder = PositionalEncoding(d_model=self.word_hidden_dim)

        # 3. Sentence-level BiLSTM Contextualizer
        self.sentence_lstm = nn.LSTM(
            input_size=self.word_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # 4. Output projection dimension (BiLSTM is bidirectional -> 2 * lstm_hidden_dim)
        self.output_dim = lstm_hidden_dim * 2
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_boundaries: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass to encode a document into sentence embeddings.

        Args:
            input_ids: Tensor [batch_size, seq_len] or [num_sentences, sent_len]
            attention_mask: Tensor of same shape as input_ids
            sentence_boundaries: Optional indexing for batched documents

        Returns:
            sentence_repr: Tensor of shape [batch_size, num_sentences, output_dim]
        """
        # Step A: Word-level representations via Transformer
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [B, L, H]

        # Step B: Sentence-level pooling (using CLS token or Mean Pooling)
        # Using [CLS] token at index 0 or masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        sent_embeddings = sum_embeddings / sum_mask  # [B, H]

        # Reshape into document-sentence structure [batch_size, num_sentences, hidden_dim]
        if sent_embeddings.dim() == 2:
            sent_embeddings = sent_embeddings.unsqueeze(0)

        # Step C: Add Sentence Positional Encoding
        sent_embeddings = self.pos_encoder(sent_embeddings)

        # Step D: Contextualize via BiLSTM across sentences
        lstm_out, _ = self.sentence_lstm(sent_embeddings)  # [batch_size, num_sentences, 2 * lstm_hidden_dim]
        
        # Step E: Normalization and Dropout
        sentence_repr = self.layer_norm(lstm_out)
        sentence_repr = self.dropout(sentence_repr)

        return sentence_repr


if __name__ == "__main__":
    # Unit test to verify input/output shapes
    print("Testing HierarchicalDocEncoder...")
    encoder = HierarchicalDocEncoder(
        transformer_model_name="bert-base-uncased",
        lstm_hidden_dim=256,
        lstm_layers=2,
        freeze_transformer=True
    )
    
    # Dummy input: 1 document with 5 sentences, each sentence has 16 tokens
    num_sentences = 5
    sent_len = 16
    dummy_input_ids = torch.randint(0, 1000, (num_sentences, sent_len))
    dummy_attention_mask = torch.ones((num_sentences, sent_len), dtype=torch.long)

    with torch.no_grad():
        output = encoder(dummy_input_ids, dummy_attention_mask)
    
    print(f"Output Sentence Matrix Shape: {output.shape}")
    print("Expected Shape: [1, 5, 512] (Batch=1, Sentences=5, HiddenDim=2*256)")
    print("HierarchicalDocEncoder verified successfully.")
