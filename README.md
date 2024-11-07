# Vision Transformer (ViT) Implementation

## Overview
This project implements a Vision Transformer (ViT) from scratch using PyTorch. The Vision Transformer model is designed for image classification tasks, using a Transformer architecture originally proposed for NLP. ViT splits an image into patches, processes these patches through multiple Transformer encoder layers, and outputs a classification prediction.

## Architecture
The Vision Transformer consists of the following key components:
1. **Patch Embedding**: Splits the image into patches and embeds each patch as a vector.
2. **Position Embedding**: Adds positional information to each patch embedding.
3. **Attention Layer**: Implements Multi-Head Self-Attention for interaction between patches.
4. **Transformer Encoder Layer**: Combines attention and feed-forward networks with residual connections.
5. **Vision Transformer**: The overall model, which includes patch and position embeddings, multiple encoder layers, and a classification head.

## Implementation Details

### 1. PatchEmbedding
- **File**: `Source/patch_embedding.py`
- **Purpose**: Divides the input image into patches and embeds each patch as a vector that can be processed by the Transformer.
- **Core Methods**:
    - `__init__`: Initializes the patch embedding layer with a convolution to handle patch splitting and embedding.
    - `forward`: Splits the input image into patches and embeds them into a fixed dimension.

### 2. PositionEmbedding
- **File**: `Source/position_embedding.py`
- **Purpose**: Adds positional information to each patch embedding, allowing the model to understand the relative positions of patches.
- **Core Methods**:
    - `__init__`: Initializes a learnable position embedding for each patch.
    - `forward`: Adds positional embeddings to the patch embeddings.

### 3. AttentionLayer
- **File**: `Source/attention_layer.py`
- **Purpose**: Implements Multi-Head Self-Attention, allowing patches to interact and exchange information.
- **Core Methods**:
    - `__init__`: Initializes the layer for multi-head attention with query, key, and value projections.
    - `forward`: Computes attention scores between patches, normalizes, and aggregates values.

### 4. TransformerEncoderLayer
- **File**: `Source/transformer_encoder_layer.py`
- **Purpose**: Encodes patch embeddings using self-attention and a feed-forward network.
- **Core Methods**:
    - `__init__`: Initializes attention and feed-forward layers with layer normalization and dropout.
    - `forward`: Applies attention and feed-forward layers with residual connections.

### 5. VisionTransformer
- **File**: `Source/vision_transformer.py`
- **Purpose**: The main Vision Transformer model for image classification.
- **Core Methods**:
    - `__init__`: Initializes all components including patch embedding, position embedding, encoder layers, and classification head.
    - `forward`: Processes an input image through the Transformer and outputs a class prediction.

## Usage
1. **Install Dependencies**
    ```bash
    pip install torch
    ```

2. **Example Usage**
    ```python
    import torch
    from Source.vision_transformer import VisionTransformer

    # Model Parameters
    img_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 10
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0
    dropout = 0.1

    # Initialize Model
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout
    )

    # Forward Pass
    input_image = torch.randn(1, 3, img_size, img_size)
    output = model(input_image)
    print(output.shape)  # Should be (1, num_classes)
    ```

## License
This project is licensed under the MIT License.

## References
- **Original Paper**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **PyTorch**: https://pytorch.org/
