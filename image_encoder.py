"""NumPy implementation of the OpenCLIP ViT-B/32 image encoder forward pass.

Input  : [B, 3, 224, 224] float32 images (already normalized with CLIP mean/std)
Output : [B, 512] L2-normalized image embeddings (inference only, no training)

Pipeline: 224x224 -> 7x7 patches of 32x32, flattened as C,H,W -> linear patch
projection (no bias) -> prepend a CLS token and add the position embedding ->
12 pre-LN Transformer blocks (attention and MLP, each with a residual, no
LayerScale) -> take the CLS row -> ln_post -> visual.proj -> L2 normalize.

Weight layout: attention and MLP weights are stored as [out, in], so they are
applied with `.T`; visual.proj is already [in, out] = [768, 512] and must not be
transposed. All intermediate tensors stay float32 to satisfy the dtype contract
checked by check.py. The LayerNorm / GELU / softmax / unit-vector helpers come
from the course starter (helpers.py) and are used unchanged.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    """Multi-head self-attention: H=12 heads, 768 features, 64 features per head."""

    def __init__(self, weights, prefix, heads):
        # prefix is e.g. "visual.transformer.resblocks.0.attn"
        self.qkv_weight = weights[prefix + '.in_proj_weight']   # [3D, D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']       # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D, D], [out, in]
        self.out_bias = weights[prefix + '.out_proj.bias']      # [D]
        self.heads = heads
        self.width = self.out_bias.shape[0]      # 768
        self.head_dim = self.width // heads      # 64

    def split_heads(self, x):
        """TODO-02: [B, N, D] -> [B, H, N, D/H] by moving the head axis forward."""
        # reshape alone only groups the features; attention needs one [N, 64]
        # matrix per head, so the head axis is moved in front of the token axis.
        batch, token_count, _ = x.shape
        head_slices = x.reshape(batch, token_count, self.heads, self.head_dim)  # [B, N, H, 64]
        return head_slices.transpose(0, 2, 1, 3)  # [B, H, N, 64]

    def __call__(self, x, causal=False):
        """Scaled dot-product attention on [B, N, 768] tokens."""
        # q, k and v are packed in one [3D, D] matrix; split the output axis.
        qkv = x @ self.qkv_weight.T + self.qkv_bias  # [B, N, 3D]
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: scaled scores. swapaxes transposes only the last two axes,
        # so each (image, head) pair keeps its own [N, N] matrix. The float32
        # cast of the scale avoids an accidental float64 promotion.
        scale = np.float32(np.sqrt(self.head_dim))
        scores = q @ np.swapaxes(k, -1, -2) / scale  # [B, H, N, N]

        # Only the text tower needs masking; images run with causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # provided softmax helper

        # TODO-03b: mix value rows, then merge the heads. Move the head axis
        # behind the token axis before the reshape, otherwise features from
        # different heads end up interleaved.
        context = coefficients @ v  # [B, H, N, 64]
        batch, _, token_count, _ = context.shape
        merged = context.transpose(0, 2, 1, 3).reshape(batch, token_count, self.width)

        # TODO-03c: output projection ([out, in] layout, so transpose the weight).
        return merged @ self.out_weight.T + self.out_bias  # [B, N, 768]


class TransformerBlock:
    """One pre-LN block: x = x + Attn(LN1(x)); x = x + MLP(LN2(x))."""

    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']    # [3072, 768]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']        # [3072]
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [768, 3072]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']      # [768]
        self.gelu = GELU()  # provided exact erf-based GELU

    def __call__(self, x, causal=False):
        # Sub-layer 1: normalized attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual connection.
        residual = x + update  # [B, N, 768]

        # Sub-layer 2: normalize again, then widen 768 -> 3072, apply GELU,
        # project back to 768.
        u = self.norm2(residual)
        hidden = self.gelu(u @ self.fc1_weight.T + self.fc1_bias)  # [B, N, 3072]
        mlp_update = hidden @ self.fc2_weight.T + self.fc2_bias   # [B, N, 768]

        # TODO-04b: second residual connection.
        return residual + mlp_update  # [B, N, 768]


class ImageEncoder:
    """The OpenCLIP ViT-B/32 visual tower: images -> [B, 512] unit vectors."""

    def __init__(self, weights):
        # conv1 is a 32x32/32 patch projection without bias, so the conv weight
        # [768, 3, 32, 32] is just a [768, 3072] linear map on flattened patches.
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)  # [1, 1, 768]
        self.position = weights['visual.positional_embedding']           # [50, 768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768, 512], already [in, out]

    def patchify(self, images):
        """TODO-01: [B, 3, 224, 224] -> [B, 49, 3072].

        Patches are visited left to right, then top to bottom, and the values
        inside a patch are flattened as C, H, W (3 * 32 * 32 = 3072). The input
        is already preprocessed, so this is only a change of memory layout.
        """
        batch = images.shape[0]
        # 1) split the spatial axes into a 7x7 patch grid of 32x32 tiles
        grid = images.reshape(batch, 3, 7, 32, 7, 32)         # [B, C, 7r, 32h, 7c, 32w]
        # 2) move the two patch axes in front of the channel axis
        patches = grid.transpose(0, 2, 4, 1, 3, 5)            # [B, 7r, 7c, C, 32h, 32w]
        # 3) 49 patch rows, each with C,H,W flattened into 3072 features
        return patches.reshape(batch, 49, 3072)               # [B, 49, 3072]

    def embed(self, images):
        """TODO-05: patch projection + CLS token + position embedding -> [B, 50, 768]."""
        patches = self.patchify(images)  # [B, 49, 3072]

        # TODO-05a: patch projection without bias, weights are [out, in].
        tokens = patches @ self.patch_weight.T  # [B, 49, 768]

        # One learned CLS row, broadcast over the batch.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))  # [B, 1, 768]

        # TODO-05b: CLS becomes token 0; position rows 1..49 follow patchify
        # order, so the concatenation order must match.
        tokens = np.concatenate([cls, tokens], axis=1)  # [B, 50, 768]
        return tokens + self.position                    # [50, 768] broadcasts over B

    def encode_tokens(self, images):
        """Provided assembly: embed, ln_pre, then run the 12 blocks."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06: [B, 3, 224, 224] -> [B, 512] unit vectors."""
        tokens = self.encode_tokens(images)  # [B, 50, 768]

        # Only the CLS row is used as the image representation.
        cls_token = tokens[:, 0, :]  # [B, 768]

        # Final LayerNorm, then the output projection. visual.proj is already
        # [in, out] = [768, 512], so it is applied without a transpose.
        output = self.post_norm(cls_token) @ self.projection  # [B, 512]

        return unit_vector(output)  # rows have length 1
