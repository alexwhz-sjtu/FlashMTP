import torch
import matplotlib.pyplot as plt
import numpy as np

def create_flashmtp_block_mask_visual(
    seq_len: int,
    anchor_positions: list,
    block_keep_mask: list,
    block_size: int,
    kvcache_window_size: int = 0,
):
    """
    生成用于可视化的 Mask 矩阵 (numpy array)
    """
    N = len(anchor_positions)
    B = 1 # Batch size 1 for visualization
    device = torch.device('cpu')
    
    # Convert inputs to tensors
    anchor_pos_tensor = torch.tensor([anchor_positions], dtype=torch.long, device=device)
    block_keep_tensor = torch.tensor([block_keep_mask], dtype=torch.bool, device=device)
    
    Q_LEN = seq_len + N * block_size
    KV_LEN = seq_len + N * block_size
    
    # Initialize mask matrix
    mask_matrix = torch.zeros(Q_LEN, KV_LEN, dtype=torch.bool)
    
    # Iterate over all Q and KV positions to apply logic manually for visualization clarity
    # Note: The original code uses a mask_mod function inside create_block_mask which 
    # broadcasts automatically. Here we simulate the result.
    
    for q_idx in range(Q_LEN):
        for kv_idx in range(KV_LEN):
            # Logic from flashmtp_mask_mod
            
            q_in_full_seq = q_idx < seq_len
            kv_in_full_seq = kv_idx < seq_len
            
            # Calculate block_id
            # If q_idx is in full seq, block_id calculation is technically out of bounds 
            # but clamped in original code. We handle logic branches first.
            
            if q_in_full_seq:
                q_block_id = -1 # Dummy
            else:
                q_block_id = (q_idx - seq_len) // block_size
                
            if kv_in_full_seq:
                kv_block_id = -1 # Dummy
            else:
                kv_block_id = (kv_idx - seq_len) // block_size
                
            # Get anchor pos for current query block
            if not q_in_full_seq:
                # Clamp ensures safety
                safe_block_id = max(0, min(N-1, q_block_id))
                anchor_pos = anchor_pos_tensor[0, safe_block_id].item()
                is_valid_block = block_keep_tensor[0, safe_block_id].item()
            else:
                anchor_pos = 0
                is_valid_block = True # Prefix is always valid
                
            # Case 1: Both in Full Sequence -> causal attention
            both_in_full = q_in_full_seq and kv_in_full_seq
            causal_mask = kv_idx <= q_idx
            
            # Case 2: Query in Block, KV in Full Sequence
            block_q_full_kv = (not q_in_full_seq) and kv_in_full_seq
            
            if kvcache_window_size > 0:
                window_start = max(0, anchor_pos - kvcache_window_size + 1)
            else:
                window_start = 0
            window_end = anchor_pos
            
            kv_in_window = (kv_idx >= window_start) and (kv_idx <= window_end)
            block_can_see_full = block_q_full_kv and kv_in_window
            
            # Case 3: Both in Block -> bidirectional within same block only
            both_in_block = (not q_in_full_seq) and (not kv_in_full_seq)
            same_block = q_block_id == kv_block_id
            block_bidirectional = both_in_block and same_block
            
            # Combine
            can_attend = (both_in_full and causal_mask) or block_can_see_full or block_bidirectional
            
            # Valid query check
            if q_in_full_seq:
                is_valid = True
            else:
                # Check if the specific block is kept
                safe_bid = max(0, min(N-1, q_block_id))
                is_valid = block_keep_tensor[0, safe_bid].item()
                
            if can_attend and is_valid:
                mask_matrix[q_idx, kv_idx] = 1
                
    return mask_matrix.numpy()

def plot_mask(mask, seq_len, n_blocks, block_size, title="FlashMTP v2 Attention Mask"):
    """
    Plot the attention mask using matplotlib.
    """
    total_len = mask.shape[0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(mask, cmap='Greys', interpolation='nearest', aspect='auto')
    
    # Draw grid lines to separate Prefix and Blocks
    # Vertical lines
    plt.axvline(x=seq_len - 0.5, color='red', linewidth=2, label='Prefix End')
    for i in range(1, n_blocks):
        x_pos = seq_len + i * block_size - 0.5
        plt.axvline(x=x_pos, color='blue', linewidth=1, linestyle='--')
        
    # Horizontal lines
    plt.axhline(y=seq_len - 0.5, color='red', linewidth=2)
    for i in range(1, n_blocks):
        y_pos = seq_len + i * block_size - 0.5
        plt.axhline(y=y_pos, color='blue', linewidth=1, linestyle='--')
        
    # Labels and Ticks
    plt.title(title, fontsize=14)
    plt.xlabel("Key/Value Index", fontsize=12)
    plt.ylabel("Query Index", fontsize=12)
    
    # Set ticks to show structure
    tick_positions = []
    tick_labels = []
    
    # Prefix start/end
    tick_positions.append(0)
    tick_labels.append("0")
    tick_positions.append(seq_len - 1)
    tick_labels.append(f"S-1\n({seq_len-1})")
    
    # Blocks
    for i in range(n_blocks):
        start = seq_len + i * block_size
        mid = start + block_size // 2
        end = start + block_size - 1
        if i == 0:
            tick_positions.append(start)
            tick_labels.append(f"B0 Start\n({start})")
        tick_positions.append(end)
        tick_labels.append(f"B{i} End\n({end})")
        
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    plt.yticks(tick_positions, tick_labels)
    
    plt.colorbar(label='Attention Allowed (1=Yes, 0=No)')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("Mask")

# --- Configuration for Demo ---
SEQ_LEN = 16          # Length of the prefix context
N_BLOCKS = 3          # Number of draft blocks
BLOCK_SIZE = 4        # Tokens per block
WINDOW_SIZE = 4       # KVCache window size

# Anchors: Positions in the prefix where each block is anchored.
# Let's say Block 0 anchors at token 15 (last token), 
# Block 1 anchors at token 15, 
# Block 2 anchors at token 10.
ANCHORS = [8, 10, 15] 
KEEP_MASK = [True, True, True]

# Generate Mask
mask_np = create_flashmtp_block_mask_visual(
    seq_len=SEQ_LEN,
    anchor_positions=ANCHORS,
    block_keep_mask=KEEP_MASK,
    block_size=BLOCK_SIZE,
    kvcache_window_size=WINDOW_SIZE
)

# Plot
plot_mask(
    mask_np, 
    SEQ_LEN, 
    N_BLOCKS, 
    BLOCK_SIZE, 
    title=f"FlashMTP Mask (S={SEQ_LEN}, N={N_BLOCKS}, Bsz={BLOCK_SIZE}, W={WINDOW_SIZE})"
)