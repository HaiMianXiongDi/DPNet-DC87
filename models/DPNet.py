import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from layers.RevIN import RevIN
from torch.nn.utils import weight_norm
















class SeqToPatch(nn.Module):

    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        individual: bool,
        patch_size: int,
        patch_stride: int,
        padding_mode: str
    ):
        super().__init__()
        self.in_channels   = in_channels
        self.seq_len       = seq_len
        self.individual    = individual  # kept for signature compatibility
        self.hidden_dim    = 128         # same constant as before

        # patch configuration
        self.patch_size    = patch_size
        self.patch_stride  = patch_stride
        self.padding_mode  = padding_mode

        # compute number of patches
        self.num_patches   = int((seq_len - patch_size) / patch_stride + 1)
        if padding_mode == "end":
            self.padding_layer = nn.ReplicationPad1d((0, patch_stride))
            self.num_patches  += 1

    def forward(self, x: Tensor) -> Tensor:
        if self.padding_mode == "end":
            x = self.padding_layer(x)                        # (B, C, L+pad)

        # → (B, C, num_patches, patch_size)
        x = x.unfold(dimension=-1,
                     size=self.patch_size,
                     step=self.patch_stride)

        # → (B, C, patch_size, num_patches)
        x = x.permute(0, 1, 3, 2).contiguous()
        return x




# ==================== FineRegulator.py  ====================
# FineRegulator corresponds to the Fine Regulator branch inside DPM (Dual Patch Mixer).
# Paper-aligned version: patch-independent (PI) decoder operating on each patch separately.
class FineRegulator(nn.Module):
    """
    Patch-Independent Fine Regulator (DPM fine branch).

    Input : x ∈ R[B, C, L]
    Output: P_f ∈ R[B, C, N, P]  (N = num_patches, P = patch_len)
    """
    def __init__(self,
                 in_channels: int,
                 input_len:   int,
                 out_len:     int,
                 individual:  bool,
                 patch_len:   int,
                 stride:      int,
                 padding:     str,
                 shared_embedding: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.c_in   = in_channels
        self.in_len = input_len
        self.out_len= out_len  # kept for signature compatibility
        self.individual = individual
        self.patch_len  = patch_len
        self.stride     = stride
        self.padding    = padding
        self.d_model    = 128

        patch_num = (input_len - patch_len) // stride + 1
        if padding == 'end':
            self.pad_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.patch_num = patch_num

        # Encoder: small MLP applied to each patch independently
        self.encoder = Encoder(in_channels, patch_len,
                               d_model=self.d_model,
                               shared_embedding=shared_embedding)

        # Decoder: PI (per-patch) linear decoder (no flatten across patches)
        self.decoder = PatchIndependentDecoder(
            individual=individual,
            n_vars=in_channels,
            d_model=self.d_model,
            patch_len=patch_len,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            [B, C, N, P]
        """
        if self.padding == 'end':
            x = self.pad_layer(x)

        # unfold into patches: [B, C, N, P]
        x = x.unfold(-1, self.patch_len, self.stride)
        # reorder to [B, C, P, N] for the shared Encoder interface
        x = x.permute(0, 1, 3, 2).contiguous()
        z = self.encoder(x)            # [B, C, d_model, N]
        p = self.decoder(z)            # [B, C, P, N]
        return p.permute(0, 1, 3, 2).contiguous()
class Encoder(nn.Module):
    """
    Two‑layer MLP encoder applied to every patch.

    Input  : x  ∈  ℝ[B, C, P, N]   (P = patch_len, N = num_patches)
    Output : z  ∈  ℝ[B, C, d_model, N]
    """
    def __init__(self,
                 c_in: int,
                 patch_len: int,
                 d_model: int,
                 *_,             # keep compatibility with extra kwargs
                 **__):
        super().__init__()
        # Shared linear layers for all variables
        self.proj_in  = nn.Linear(patch_len, d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.act      = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, P, N]  →  [B, N, C, P]
        x = x.permute(0, 3, 1, 2)

        # Single shared projection applied to all channels
        x = self.proj_in(x)
        x = self.act(x)
        x = self.proj_out(x)                       # [B, N, C, d_model]

        # Re‑order to [B, C, d_model, N]
        x = x.transpose(1, 2).permute(0, 1, 3, 2).contiguous()
        return x


# ==================== pi_decoder.py ====================
class PatchIndependentDecoder(nn.Module):
    """
    Patch-Independent (PI) decoder for the Fine Regulator.

    It maps each patch embedding z_n ∈ R^{d_model} to a length-P vector independently,
    i.e., it does NOT flatten across patches.

    Input : z ∈ R[B, C, d_model, N]
    Output: p ∈ R[B, C, P, N]
    """
    def __init__(self,
                 individual: bool,
                 n_vars: int,
                 d_model: int,
                 patch_len: int,
                 dropout: float = 0.0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if individual:
            self.mappers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, patch_len),
                    nn.Dropout(dropout)
                ) for _ in range(n_vars)
            ])
        else:
            self.linear = nn.Linear(d_model, patch_len)
            self.dropout = nn.Dropout(dropout)

    def forward(self, z: Tensor) -> Tensor:
        # z: [B, C, d_model, N]
        if self.individual:
            outs = []
            for i in range(z.size(1)):
                zi = z[:, i]                 # [B, d_model, N]
                zi = zi.permute(0, 2, 1)     # [B, N, d_model]
                pi = self.mappers[i](zi)     # [B, N, P]
                pi = pi.permute(0, 2, 1)     # [B, P, N]
                outs.append(pi)
            return torch.stack(outs, dim=1)  # [B, C, P, N]

        z_perm = z.permute(0, 3, 1, 2)       # [B, N, C, d_model]
        out = self.linear(z_perm)            # [B, N, C, P]
        out = self.dropout(out)
        return out.permute(0, 2, 3, 1).contiguous()  # [B, C, P, N]
# ==================== decoder.py ====================
class Decoder(nn.Module):
    """
    Flatten [B, C, d_model, N] and map to the prediction window.
    """
    def __init__(self,
                 individual: bool,
                 n_vars:     int,
                 in_dim:     int,
                 target_len: int,
                 dropout:    float = 0.0):
        super().__init__()
        self.individual = individual

        if individual:
            self.mappers = nn.ModuleList([
                nn.Sequential(
                    nn.Flatten(start_dim=-2),
                    nn.Linear(in_dim, target_len),
                    nn.Dropout(dropout)
                ) for _ in range(n_vars)
            ])
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear  = nn.Linear(in_dim, target_len)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, d_model, N]
        if self.individual:
            out = [self.mappers[i](x[:, i]) for i in range(x.size(1))]
            return torch.stack(out, dim=1)          # [B, C, target_len]
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)

# ==================== CoarseRegulator.py ====================
# CoarseRegulator corresponds to the Coarse Regulator branch inside DPM
# (Dual Patch Mixer) in the paper.
class CoarseRegulator(nn.Module):
    """
    Upsample‑and‑Refine block used in cross‑scale fusion.
    Input  : [B, C, P_d, N_d]  (flattened internally)
    Output : [B, C, P_c, N_c]  (shape kept flat; caller reshapes)
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=-2),          # flatten last two dims
            nn.Linear(in_dim, out_dim),        # project to target length
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),       # fine‑tune
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [B, C, P_d, N_d]
        return self.net(x)                     # [B, C, out_dim]






class MLPblc(nn.Module):
    """
    input:  x1, x2  --  [B, C, P, N]
    output: x_fuse  --  [B, C, P, N]

    Note:
    The 1×1 Conv2d layers used here are mathematically equivalent to
    position-wise Linear projections (shared across all spatial positions).
    They are adopted only for implementation convenience.
    """

    def __init__(self, channels, dropout=0.1, hidden_ratio=0.5):
        super().__init__()
        hidden = max(8, int(channels * 2 * hidden_ratio))  # 2C → hidden → 2C
        self.mlp = nn.Sequential(
            nn.Conv2d(2*channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, 2*channels, kernel_size=1),
            nn.Dropout(dropout)
        )

        self.gate = nn.Sequential(
            nn.Conv2d(2*channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):  
        z = torch.cat([x1, x2], dim=1)      # [B, 2C, P, N]
        z = self.mlp(z)                     # [B, 2C, P, N]
        g = self.gate(z)                    # [B, C,  P, N] (0~1)
        x_fuse = g * x1 + (1. - g) * x2
        return x_fuse

class MLPBlock(nn.Module):
    """
    A simple MLP block: Linear -> GELU -> Dropout
    Input shape: [..., dim]
    Output shape: [..., dim] (the same as input)
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x shape is (..., dim)
        out = self.linear(x)
        out = self.act(out)
        out = self.drop(out)
        return out


# ==================== Predictor.py ====================
class Predictor(nn.Module):
    """
    input:  [B, C, P, N]    (P = patch_len, N = num_patch)
    output:  [B, C, out_len]
    """
    def __init__(self,
                 in_channels: int,
                 input_len:   int,      
                 out_len:     int,
                 individual:  bool,
                 patch_len:   int,
                 stride:      int,
                 padding:     str,
                 shared_embedding: bool = True,
                 dropout: float = 0.0):

        super().__init__()
        d_model   = 128
        patch_num = (input_len - patch_len) // stride + 1
        if padding == 'end':
            patch_num += 1

        self.encoder = Encoder(c_in=in_channels,
                               patch_len=patch_len,
                               d_model=d_model,
                               shared_embedding=shared_embedding)

        self.decoder = Decoder(individual=individual,
                               n_vars=in_channels,
                               in_dim=d_model * patch_num,
                               target_len=out_len,
                               dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:          # x:[B,C,P,N]
        x = self.encoder(x)
        return self.decoder(x)                      # x:[B,C,out_len]

  
  
  
  
  
  
   
   
class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()

        self.input_channels = configs.enc_in
        self.input_len = configs.seq_len
        self.out_len = configs.pred_len
        self.individual = configs.individual
        self.stage_num = configs.stage_num
        self.stage_pool_kernel = configs.stage_pool_kernel
        self.stage_pool_stride = configs.stage_pool_stride
        self.stage_pool_padding = configs.stage_pool_padding   
        self.revin_layer = RevIN(self.input_channels, affine=True, subtract_last=False)

       
        


    
        
        self.configs = configs
        # 1) Pre-calculate the sequence lengths after multi-scale downsampling (similar to the original down_in logic)
        #    Used for performing stage_num layers of downsampling on current_sequence ([B, C, L])
        self.ms_down_in = []
        cur_len = self.input_len
        self.ms_down_in.append(cur_len)
        for i in range(self.stage_num - 1):
            next_len = int((cur_len + 2*self.stage_pool_padding - self.stage_pool_kernel)
                        / self.stage_pool_stride + 1)
            self.ms_down_in.append(next_len)
            cur_len = next_len

        # 2) Downsampling modules (AvgPool1d), using the same parameters as before
        self.downsamplers_extra = nn.ModuleList()
        for i in range(self.stage_num - 1):
            self.downsamplers_extra.append(
                nn.AvgPool1d(
                    kernel_size=self.stage_pool_kernel,
                    stride=self.stage_pool_stride,
                    padding=self.stage_pool_padding
                )
            )


        # Fine Regulators (DPM fine branch) for each pyramid scale
        # Note: FineRegulator returns patch-wise outputs directly (PI per-patch decoding),
        # so we do NOT unfold it again outside.
        self.fine_regulators = nn.ModuleList()
        for i in range(self.stage_num):
            fine_reg = FineRegulator(
                in_channels=self.input_channels,
                input_len=self.ms_down_in[i],
                out_len=self.ms_down_in[i],
                individual=self.individual,
                patch_len=self.configs.patch_len,
                stride=self.configs.stride,
                padding=self.configs.padding_patch,
                shared_embedding=True,
                dropout=self.configs.head_dropout,
            )
            self.fine_regulators.append(fine_reg)

        self.predictors = nn.ModuleList()
        for i in range(self.stage_num):
            decoder = Predictor(
                in_channels  = self.input_channels,
                input_len    = self.ms_down_in[i],
                out_len      = self.out_len,
                individual   = self.individual,
                patch_len    = self.configs.patch_len,
                stride       = self.configs.stride,
                padding      = self.configs.padding_patch,
                shared_embedding = True,
                dropout          = self.configs.head_dropout
            )
            self.predictors.append(decoder)

 
        

        # ── MLP blocks ──
        self.mlp_temps = nn.ModuleList()
        self.mlp_chans = nn.ModuleList()
        self.mlp_blcs = nn.ModuleList()
        # Coarse Regulators (DPM coarse branch) between adjacent pyramid scales
        self.coarse_regulators = nn.ModuleList()

        for i_stage in range(self.stage_num):
            # Calculate the total patch length PN_i for this stage
            pn_i = self.configs.patch_len * int(
                (self.ms_down_in[i_stage] - self.configs.patch_len) / self.configs.stride + 1 +
                (1 if self.configs.padding_patch == 'end' else 0)
            )  
            self.mlp_temps.append(
                MLPBlock(dim=pn_i, dropout=self.configs.head_dropout)
            )
            self.mlp_chans.append(
                MLPBlock(dim=self.input_channels, dropout=self.configs.head_dropout)
            )
            self.mlp_blcs.append(MLPblc(channels=self.input_channels,
                                          dropout=self.configs.head_dropout))
            
            
        self.patch_nums = []
        for i_len in self.ms_down_in:
            p_num = int((i_len - self.configs.patch_len) / self.configs.stride + 1)
            if self.configs.padding_patch == 'end':
                p_num += 1
            self.patch_nums.append(p_num)

        for i in range(self.stage_num - 1):
            in_dim  = self.patch_nums[i+1] * self.configs.patch_len   # (i+1) 
            out_dim = self.patch_nums[i]   * self.configs.patch_len   # (i)   
            self.coarse_regulators.append(
                CoarseRegulator(
                    in_dim  = in_dim,
                    out_dim = out_dim,
                    dropout = self.configs.head_dropout
                )
            )
            
            


        
                
        #-------------------############------------------------------############-------------#
  
    def forward(self, x):
        
        debug = getattr(self, 'record_debug', False)
        x_norm    = self.revin_layer(x, 'norm')        # [B, L, C]
        current_sequence = x_norm.permute(0, 2, 1)            # [B, C, L]
        B, C, L = current_sequence.shape


        ########################################
        # 1) downsampling => stages
        ########################################
        stages = [current_sequence]  
        for i in range(self.stage_num - 1):
            x_in = stages[-1]                          # [B, C, length_i]
            bc = x_in.reshape(B*C, 1, -1)              # => [B*C,1,length_i]
            ds = self.downsamplers_extra[i](bc)        # => [B*C,1,length_(i+1)]
            ds = ds.reshape(B, C, -1)                  # => [B, C, length_(i+1)]
            stages.append(ds)

        ########################################
        # 2) unfold + FineRegulator => fine-regulated patches
        ########################################
        source_patched_stages = []
        fine_regulated_patches = []
        for i in range(self.stage_num):
            stage_x = stages[i]  # [B, C, length_i]
            # (A) unfold => [B, C, num_patches, patch_len]
            patch_len = self.configs.patch_len
            stride_   = self.configs.stride
            if self.configs.padding_patch == 'end':
                pad_stage_x = F.pad(stage_x, (0, stride_), mode='replicate')
            else:
                pad_stage_x = stage_x
            x_unfold = pad_stage_x.unfold(dimension=-1, size=patch_len, step=stride_)


            source_patched_stages.append(x_unfold)  # [B, C, num_patches, patch_len]    P(s)  P(s+1)....
            

            # FineRegulator returns patch-wise outputs directly: [B, C, num_patches, patch_len]
            fine_regulated_seq = self.fine_regulators[i](stage_x)
            fine_regulated_patches.append(fine_regulated_seq)
            



       ########################################
        # 3) DPM (Dual Patch Mixer) mixing
        ########################################
        for i in reversed(range(self.stage_num - 1)):
            deeper_patched = source_patched_stages[i + 1]   # [B,C,np_d,p_d]         P(s+1)
            cur_patched    = source_patched_stages[i]       # [B,C,np_c,p_c]         P(s)
            cur_fine       = fine_regulated_patches[i]      # [B,C,np_c,p_c]         P_f(s)


            B_, C_, npd, pld = deeper_patched.shape
            B_, C_, npc, plc = cur_patched.shape
            up_flat  = self.coarse_regulators[i](deeper_patched)        # [B, C, out_dim]
            coarse_regulated = up_flat.view(B_, C_, npc, plc).contiguous()  # [B,C,np_c,p_c]  P_c(s+1->s)

            mixed = coarse_regulated + cur_fine            # P_mix(s) = P_c(s+1->s) + P_f(s)
            source_patched_stages[i] = mixed
            




        fused_patched_stages = source_patched_stages   # [B, C, num_patches, patch_len]
        ########################################
        # 4) Patch Relation Reconstructor (PRR) + Predictors
        # Note: Dimensions N and P are always flattened together (N*P), thus their order doesn't affect results.
        # Here, input shape is set to [B,C,P,N] just for convenience of function calls and data handling.
        ########################################
        predictions = []
        
        for i in range(self.stage_num):
            x_in = fused_patched_stages[i].permute(0, 1, 3, 2).contiguous()  # [B,C,P,N]
            B_, C_, P_, N_ = x_in.shape
            PN_ = P_ * N_

            # —— MLPtemp —— #
            x_flat   = x_in.reshape(B_, C_, PN_).contiguous()          # [B,C,PN_]     #Flat
            out1     = self.mlp_temps[i](x_flat.view(B_ * C_, PN_))  # [B*C,PN_]       # —— MLPtemp —— #
            out1     = out1.view(B_, C_, PN_)
            x1       = x_flat + out1                                                   #Residual
            x1_4d    = x1.view(B_, C_, P_, N_)                         # [B,C,P,N]     #Unflat


             # —— MLPchan —— #
            x1_T     = x_flat.permute(0, 2, 1).contiguous()               # [B,PN_,C]      #Swap
            out2     = self.mlp_chans[i](x1_T.view(B_ * PN_, C_))                      # —— MLPchan —— #
            out2     = out2.view(B_, PN_, C_)
            x2       = x1_T + out2                                                     #Residual
            x2_C     = x2.view(B_, P_, N_, C_).permute(0, 3, 1, 2)    # [B,C,P,N]      #Swap -> Unflat


            # —— MLPblc —— #
            x_fuse   = self.mlp_blcs[i](x1_4d, x2_C)             # [B,C,P,N]          # —— MLPblc —— #
            x_enhanced = x_fuse + x_in                                                 #Residual
            
    
            # —— Predictors —— #
            pred_i = self.predictors[i](x_enhanced)         # [B,C,out_len]
            predictions.append(pred_i)
            
        ########################################
        # 5) => final_prediction  (sum of all predictions)
        ########################################
        final_prediction = sum(predictions)  # => [B, C, self.out_len]
                
        
    
        #-------------------############------------------------------############-------------#

        final_prediction = final_prediction.permute(0, 2, 1)  # [Batch, Seq_len, Channels]
        final_prediction = self.revin_layer(final_prediction, 'denorm')
        

        return final_prediction






    

    

    

    