import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

#. utils
from TopoDiff.utils.debug import log_var
from TopoDiff.utils.structure import reconstruct_backbone_position, reconstruct_backbone_position_without_torsion

from myopenfold.utils.rigid_utils import Rotation, Rigid
from myopenfold.utils.tensor_utils import dict_multimap

#. np
from myopenfold.np import residue_constants as rc

class StructureModuleHelper(nn.Module):

    def __init__(self):
        super(StructureModuleHelper, self).__init__()

    def _prepare_backbone_constant(self, float_dtype, device):
        if not hasattr(self, 'restype_backbone_rigid_group_default_frame'):
            restype_backbone_rigid_group_default_frame = rc.restype_rigid_group_default_frame[:, 3]
            restype_backbone_rigid_group_default_frame[20] = rc.average_rigid_group_default_frame[0, 3]
            self.register_buffer('restype_backbone_rigid_group_default_frame', 
                                 torch.tensor(restype_backbone_rigid_group_default_frame,
                                              dtype=float_dtype, device=device,
                                              requires_grad=False))
            
        if not hasattr(self, 'restype_backbone_rigid_group_positions'):
            restype_backbone_rigid_group_positions = rc.restype_atom14_rigid_group_positions[:, :5]
            restype_backbone_rigid_group_positions[20] = rc.average_atom14_rigid_group_positions[0, :5]
            self.register_buffer('restype_backbone_rigid_group_positions',
                                torch.tensor(restype_backbone_rigid_group_positions,
                                            dtype=float_dtype, device=device,
                                            requires_grad=False))
            
        return
    

    def to_rigid(self, frame_pred, intype: Union['rigid', 'tensor_4x4', 'tensor_7', 'tuple'] = 'rigid'):
        if intype == 'rigid':
            return frame_pred
        elif intype == 'tensor_4x4':
            return Rigid.from_tensor_4x4(frame_pred)
        elif intype == 'tensor_7':
            return Rigid.from_tensor_7(frame_pred)
        elif intype == 'tuple':
            return Rigid.from_tensor_RT(*frame_pred)
        else:
            raise NotImplementedError(f'Unknown intype {intype}')
        
    
    def reconstruct_backbone_position_wrap(self, frame_pred, torsion_sin_cos, torsion_index = 0, seq_type = None,
                                           intype: Union['rigid', 'tensor_4x4', 'tensor_7', 'tuple'] = 'rigid',
                                           include_CB = False):
        """Wrap of `reconstruct_backbone_position`

        Args:
            frame_pred: [..., N_res]
                Rigid objects
            torsion_sin_cos: [..., N_res, N_torsion, 2]
                Torsion angles in sin and cos form
            torsion_index (int)
                The index of Psi angle
            seq_type: [..., N_res]
                The sequence type of each residue.
        """
        frame_pred = self.to_rigid(frame_pred, intype)

        self._prepare_backbone_constant(torsion_sin_cos.dtype, torsion_sin_cos.device)

        coord_recon = reconstruct_backbone_position(
            restype_backbone_rigid_group_default_frame=self.restype_backbone_rigid_group_default_frame,
            restype_backbone_rigid_group_positions=self.restype_backbone_rigid_group_positions,
            frame_pred=frame_pred,
            torsion_sin_cos=torsion_sin_cos,
            torsion_index=torsion_index,
            seq_type=seq_type,
            include_CB=include_CB
            )

        return coord_recon
    

    def reconstruct_backbone_position_without_torsion_wrap(self, frame_pred, seq_type = None, 
                                                           intype: Union['rigid', 'tensor_4x4', 'tensor_7', 'tuple'] = 'rigid'):
        frame_pred = self.to_rigid(frame_pred, intype)

        self._prepare_backbone_constant(frame_pred.get_trans().dtype, frame_pred.get_trans().device)

        coord_recon = reconstruct_backbone_position_without_torsion(
            restype_backbone_rigid_group_default_frame=self.restype_backbone_rigid_group_default_frame,
            restype_backbone_rigid_group_positions=self.restype_backbone_rigid_group_positions,
            frame_pred=frame_pred,
            seq_type=seq_type
            )

        return coord_recon






            

    

        












        
