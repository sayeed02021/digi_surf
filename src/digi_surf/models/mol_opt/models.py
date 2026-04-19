import torch
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoder
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Implements sinusoidal positional encoding from
        'Attention Is All You Need' (Vaswani et al. 2017).

        Args:
            d_model: embedding dimension
            max_len: maximum sequence length supported
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class PropertyEmbedder(nn.Module):
    """
    Encodes the property values of source and target into embeddings
    
    Shapes: 
    input: (batch, n_prop, 3) (3 parameters are source value, output value, and difference)
    output: (batch, n_prop, d)
    """
    def __init__(self, d, n_prop):
        super().__init__()
        # self.dim = d
        self.model = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, d)
        )
        self.n_prop = n_prop
        self.id_embedding = nn.Embedding(n_prop, d)

    def forward(self, x, prop_id = None):
        """
        x: (B,n_prop,3)
        prop_id: (B,n_prop) Ex for B=1 prop_id = [0,1,2]
        
        prop_id: Useful if n_prop!=all properties model was trained on
        Ex: If model was trained on pCMC, AW_ST_CMC, Area_min, inference only supplies pCMC and Area_min then 
        prop_id = [0,2] 
        """
        B, n_prop, _ = x.shape
        if prop_id is None:
            if x.size(1)<self.n_prop:
                raise ValueError(f"Expected {self.n_prop} property values, but received {x.size(1)} properties. Please provide prop_id as well or more property values")
            elif x.size(1)>self.n_prop:
                raise ValueError(f"Expected {self.n_prop} property values, but received {x.size(1)} properties. Please provide less number of property values")
            else:
                prop_id = torch.arange(self.n_prop, dtype=torch.long, device=x.device).repeat(B,1) # (B,self.n_prop)

        x = x.reshape(B*n_prop, 3)
        x = self.model(x)
        x = x.reshape(B, n_prop, -1)
        return x + self.id_embedding(prop_id)
    

    


class EncoderDecoderTrfm(nn.Module):
    """
    Arguments:
    d_model: embedding dimension
    nhead: number of attn heads
    dim_feedforward: output dim of feedforward network
    N: number of encoder layers
    max_seq_len: maximum length of sequence
    num_embeddings: number of tokens that transformer takes
    embedding_dim: dimension of token embeddings
    padding_idx: idx of padded_dimension
    
    """

    def __init__(
        self,
        num_embeddings,
        n_prop,
        padding_idx=None,
        d_model=512,
        nhead=8,
        dim_feedforward=512,
        num_layers=6,
        max_seq_len=128,
        
    ):
        super().__init__()
        self.n_prop = n_prop
        
        self.embedder = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        encoder_layer = TransformerEncoderLayer(
            d_model = d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=dim_feedforward
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first = True,
            dim_feedforward=dim_feedforward
        )
        self.padding_idx = padding_idx
    
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
    
        self.prop_encoder = PropertyEmbedder(d_model, n_prop=self.n_prop)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        self.output_proj = nn.Linear(d_model, num_embeddings)

    def forward(self, src, tgt, prop=None, prop_id = None):
        """
        src: batch_size, source_len(List of token indices with padding as well, so max_seq_len)
        tgt: batch_size, target_len(List of token indices with padding idx as well, so max_seq_len)
        prop: batch_size, 3 (property changes)
        
        """
        B, src_len = src.shape
        _, tgt_len = tgt.shape
        
        

        src_emb = self.embedder(src)
        tgt_emb = self.embedder(tgt)
        
        src_key_padding_mask = (src==self.padding_idx) # B, S
        tgt_key_padding_mask = (tgt==self.padding_idx)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(src.device)

        if prop is None:
            final_src_emb = src_emb
        else:
            n_prop = prop.size(1)
            if n_prop!=self.n_prop and prop_id is None:
                raise ValueError(f'Properties in input({n_prop}) do not match number of properties model can handle({self.n_prop})')
            if prop_id is None:
                prop_id = torch.arange(self.n_prop, dtype=torch.long, device=src.device).repeat(B,1) # (B,self.n_prop)
            
            prop_emb = self.prop_encoder(prop, prop_id) # (B,n_prop,d)
            prop_mask = torch.zeros(B, n_prop, dtype=torch.bool, device=src.device)
            
            src_key_padding_mask = torch.cat([prop_mask, src_key_padding_mask[:, :-n_prop]], dim=1)
            final_src_emb = torch.cat([prop_emb, src_emb[:,:-n_prop,:]], dim=1) # (B,S,d)

        final_src_emb = self.pos_encoder(final_src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        memory = self.encoder(
            src = final_src_emb,
            src_key_padding_mask=src_key_padding_mask
        )

        decoder_output = self.decoder(
            tgt = tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask = src_key_padding_mask
            
        )
        logits = self.output_proj(decoder_output)
        return logits
            
    
            
    