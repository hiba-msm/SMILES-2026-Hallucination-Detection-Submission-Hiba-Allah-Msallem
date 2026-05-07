"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # Last-token extraction from layers spanning the model's depth.
    # The last token carries the strongest hallucination signal as it
    # reflects the model's final commitment to its answer.
    # ------------------------------------------------------------------

    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    # Embedding (0), Early (6), Mid (12), Late (18), Final (24)
    layer_indices = [0, 6, 12, 18, 24]
    layers_to_pool = hidden_states[layer_indices, last_pos, :]  # (5, hidden_dim)
    feature = layers_to_pool.flatten()                          # (5 * hidden_dim,)

    return feature
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # Topological & Geometric Feature Extraction
    # We treat the transformer's layer stack as a discrete trajectory
    # through representation space and compute geometric invariants.
    # ------------------------------------------------------------------

    n_layers = hidden_states.shape[0]  # typically 25 (embedding + 24 layers)
    mask = attention_mask.bool()

    # 1. Layer-wise L2 norms of mean-pooled representations
    #    ("magnitude trajectory" — how the activation scale evolves)
    norms = []
    mean_pooled = []
    for li in range(n_layers):
        mp = hidden_states[li][mask].mean(dim=0)  # (hidden_dim,)
        mean_pooled.append(mp)
        norms.append(torch.norm(mp, p=2))
    norms_tensor = torch.stack(norms)  # (n_layers,)

    # 2. Consecutive cosine similarities ("representation drift")
    #    Measures angular change between adjacent layers
    drifts = []
    for i in range(n_layers - 1):
        sim = F.cosine_similarity(mean_pooled[i].unsqueeze(0),
                                  mean_pooled[i + 1].unsqueeze(0))[0]
        drifts.append(sim)
    drifts_tensor = torch.stack(drifts)  # (n_layers - 1,)

    # 3. Activation variance per layer ("confidence trajectory")
    variances = []
    for li in range(n_layers):
        variances.append(torch.var(mean_pooled[li]))
    variances_tensor = torch.stack(variances)  # (n_layers,)

    # 4. Summary statistics of the trajectories themselves
    #    These are second-order topological features.
    trajectory_stats = torch.tensor([
        norms_tensor.mean(),
        norms_tensor.std(),
        drifts_tensor.mean(),
        drifts_tensor.std(),
        drifts_tensor.min(),       # minimum drift = most stable transition
        variances_tensor.mean(),
        variances_tensor.std(),
    ], device=hidden_states.device)

    return torch.cat([norms_tensor, drifts_tensor, variances_tensor, trajectory_stats], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    use_geometric = True # override to always extract geometric features
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
