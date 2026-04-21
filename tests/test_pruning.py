import pytest
import torch
import torch.nn as nn
import numpy as np

from moshilite.pruning.depth_pruning import prune_layers
from moshilite.pruning.layer_collapse import collapse_layers
from moshilite.pruning.head_pruning import prune_heads
from moshilite.pruning.ffn_pruning import prune_ffn_channels
from moshilite.analysis.head_importance import HeadImportanceResult
from moshilite.analysis.ffn_importance import FFNImportanceResult


# --- Mock Moshi Model ---

class MockAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.embed_dim = 32
        
        self.q_proj = nn.Linear(32, 32)
        self.k_proj = nn.Linear(32, 32)
        self.v_proj = nn.Linear(32, 32)
        self.out_proj = nn.Linear(32, 32)
        
        # Test out_projs list too
        self.out_projs = nn.ModuleList([nn.Linear(32, 32)])

class MockGating(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_in = nn.Linear(32, 64)
        self.linear = nn.Linear(32, 64)

class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockAttn()
        self.gating = MockGating()
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 32)

class MockTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 10 layers
        self.layers = nn.ModuleList([MockLayer() for _ in range(10)])

class MockMoshiLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = MockTransformer()


# --- Tests ---

def test_depth_pruning():
    model = MockMoshiLMModel()
    assert len(model.transformer.layers) == 10
    
    # Prune layers 2, 5, 8
    model = prune_layers(model, [2, 5, 8])
    assert len(model.transformer.layers) == 7

def test_layer_collapse():
    model = MockMoshiLMModel()
    
    # Store weights before collapse
    original_layer_1_w = model.transformer.layers[1].self_attn.q_proj.weight.clone()
    original_layer_2_w = model.transformer.layers[2].self_attn.q_proj.weight.clone()
    
    # Collapse layer 2. It should merge into layer 1 (closest preceding)
    model = collapse_layers(model, [2])
    
    assert len(model.transformer.layers) == 9
    
    # New layer 1 weight should be average of old layer 1 and 2
    new_layer_1_w = model.transformer.layers[1].self_attn.q_proj.weight
    expected_w = (original_layer_1_w + original_layer_2_w) / 2.0
    
    assert torch.allclose(new_layer_1_w, expected_w)

def test_head_pruning_uniform():
    model = MockMoshiLMModel()
    
    # Mock importance result: all heads scored equally
    scores = np.zeros((10, 4))
    ranked = [(l, h, 0.0) for l in range(10) for h in range(4)]
    hr = HeadImportanceResult(scores, ranked)
    
    # Prune 50% uniformly -> 2 out of 4 heads should be kept
    model = prune_heads(model, hr, max_pct=0.5, mode="uniform")
    
    for layer in model.transformer.layers:
        assert layer.self_attn.num_heads == 2
        assert layer.self_attn.embed_dim == 16
        
        # Check q_proj shape: in=32, out=16
        assert layer.self_attn.q_proj.in_features == 32
        assert layer.self_attn.q_proj.out_features == 16
        
        # Check out_proj shape: in=16, out=32
        assert layer.self_attn.out_proj.in_features == 16
        assert layer.self_attn.out_proj.out_features == 32

def test_head_pruning_non_uniform():
    model = MockMoshiLMModel()
    scores = np.zeros((10, 4))
    ranked = [(l, h, 0.0) for l in range(10) for h in range(4)]
    hr = HeadImportanceResult(scores, ranked)
    
    # Array of layer similarities where some are much higher than others
    # mean is 1.0. Layer 0 has sim 0.0 -> prune ratio 0
    # Layer 1 has sim 2.0 -> prune ratio 0.5 * 2 = 1.0 -> clipped to 0.9 -> prunes 3 heads
    sims = np.ones(10)
    sims[0] = 0.0
    sims[1] = 2.0 
    
    model = prune_heads(model, hr, max_pct=0.5, mode="non_uniform", cosine_similarities=sims)
    
    # Layer 0 keeps all 4
    assert model.transformer.layers[0].self_attn.num_heads == 4
    # Layer 1 prunes ~90% of 4 = 3 dropped -> keeps 1
    assert model.transformer.layers[1].self_attn.num_heads == 1

def test_ffn_pruning_uniform():
    model = MockMoshiLMModel()
    
    imp = {}
    var_rat = {}
    ranked = {}
    for i in range(10):
        imp[i] = np.zeros(64)
        var_rat[i] = np.zeros(1)
        ranked[i] = list(range(64))
        
    fr = FFNImportanceResult(imp, var_rat, ranked)
    
    model = prune_ffn_channels(model, fr, max_pct=0.25, mode="uniform")
    
    for layer in model.transformer.layers:
        # linear_in projects 32 -> 64. 25% pruned -> 48 features
        assert layer.gating.linear_in.in_features == 32
        assert layer.gating.linear_in.out_features == 48
        
        assert layer.linear1.out_features == 48
        assert layer.linear2.in_features == 48
