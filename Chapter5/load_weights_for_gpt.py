import numpy as np
import torch


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.transformer_blocks[b].attention.W_query.weight = assign(
            gpt.transformer_blocks[b].attention.W_query.weight, q_w.T
        )
        gpt.transformer_blocks[b].attention.W_key.weight = assign(
            gpt.transformer_blocks[b].attention.W_key.weight, k_w.T
        )
        gpt.transformer_blocks[b].attention.W_value.weight = assign(
            gpt.transformer_blocks[b].attention.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.transformer_blocks[b].attention.W_query.bias = assign(
            gpt.transformer_blocks[b].attention.W_query.bias, q_b
        )
        gpt.transformer_blocks[b].attention.W_key.bias = assign(
            gpt.transformer_blocks[b].attention.W_key.bias, k_b
        )
        gpt.transformer_blocks[b].attention.W_value.bias = assign(
            gpt.transformer_blocks[b].attention.W_value.bias, v_b
        )

        gpt.transformer_blocks[b].attention.out_proj.weight = assign(
            gpt.transformer_blocks[b].attention.out_proj.weight,
            (params["blocks"][b]["attn"]["c_proj"])["w"].T,
        )
        gpt.transformer_blocks[b].attention.out_proj.bias = assign(
            gpt.transformer_blocks[b].attention.out_proj.bias,
            (params["blocks"][b]["attn"]["c_proj"])["b"],
        )

        gpt.transformer_blocks[b].feed_forward.layers[0].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].weight,
            (params["blocks"][b]["mlp"]["c_fc"])["w"].T,
        )
        gpt.transformer_blocks[b].feed_forward.layers[0].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].bias,
            (params["blocks"][b]["mlp"]["c_fc"])["b"],
        )
        gpt.transformer_blocks[b].feed_forward.layers[2].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].weight,
            (params["blocks"][b]["mlp"]["c_proj"])["w"].T,
        )
        gpt.transformer_blocks[b].feed_forward.layers[2].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].bias,
            (params["blocks"][b]["mlp"]["c_proj"])["b"],
        )

        gpt.transformer_blocks[b].ln1.scale = assign(
            gpt.transformer_blocks[b].ln1.scale,
            (params["blocks"][b]["ln_1"])["g"],
        )
        gpt.transformer_blocks[b].ln1.shift = assign(
            gpt.transformer_blocks[b].ln1.shift,
            (params["blocks"][b]["ln_1"])["b"]
        )
        gpt.transformer_blocks[b].ln2.scale = assign(
            gpt.transformer_blocks[b].ln2.scale,
            (params["blocks"][b]["ln_2"])["g"]
        )
        gpt.transformer_blocks[b].ln2.shift = assign(
            gpt.transformer_blocks[b].ln2.shift,
            (params["blocks"][b]["ln_2"])["b"]
        )

    gpt.final_norm.scale = assign(
        gpt.final_norm.scale, params["g"]
    )
    gpt.final_norm.shift = assign(
        gpt.final_norm.shift, params["b"]
    )
    # Out head reuses the weights from the token embedding layer
    gpt.out_head.weight = assign(
        gpt.out_head.weight, params["wte"]
    )