from typing import Any, Sequence, Mapping, Optional
import string
import os

import numpy as np
import torch
import torch.nn as nn

from myopenfold.utils.rigid_utils import Rotation, Rigid
from myopenfold.utils.tensor_utils import batched_gather

from myopenfold.np import residue_constants as rc
from myopenfold.np import protein

import py3Dmol

class StructureBuilder(nn.Module):
    def __init__(self):
        super(StructureBuilder, self).__init__()

        self._init()

    def _init(self):
        restype_atom37_to_atom14 = []
        restype_atom37_mask = []
        restype_atom14_to_atom37 = []
        restype_atom14_mask = []

        for rt in rc.restypes:
            #. list of 14 atom names for this restype
            #. ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', '']
            atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]

            #. [0, 1, 2, 4, 3, 6, 7, 0, 0, 0, 0, 0, 0, 0]
            restype_atom14_to_atom37.append(
                [(rc.atom_order[name] if name else 0) for name in atom_names]
            )

            #. list of maske for atom existence for this restype
            #. [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
            restype_atom14_mask.append(
                [(1.0 if name else 0.0) for name in atom_names]
            )

            #. {atom name : index of atom-37}
            #. {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6, '': 13}
            atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}

            #. list of index of atom-37 for this restype
            #. [0, 1, 2, 4, 3, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            restype_atom37_to_atom14.append(
                [
                    atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0 for name in rc.atom_types
                ]
            )

            #. [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            restype_atom37_mask.append(
                [
                    1 if name in atom_name_to_idx14 else 0 for name in rc.atom_types
                ]
            )
        restype_atom14_to_atom37.append([0] * 14)
        restype_atom14_mask.append([0.0] * 14)
        restype_atom37_to_atom14.append([0] * 37)
        restype_atom37_mask.append([0.0] * 37)

        restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int64)
        restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.int32)
        restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int64)
        restype_atom37_mask = np.array(restype_atom37_mask, dtype=np.int32)

        self.register_buffer('restype_atom37_to_atom14', torch.from_numpy(restype_atom37_to_atom14))
        self.register_buffer('restype_atom37_mask', torch.from_numpy(restype_atom37_mask))
        self.register_buffer('restype_atom14_to_atom37', torch.from_numpy(restype_atom14_to_atom37))
        self.register_buffer('restype_atom14_mask', torch.from_numpy(restype_atom14_mask))

        # restype_rigid_group_default_frame, restype_atom14_to_rigid_group, restype_atom14_mask, restype_atom14_rigid_group_positions
        self.register_buffer('restype_rigid_group_default_frame', torch.from_numpy(np.concatenate([rc.restype_rigid_group_default_frame, rc.average_rigid_group_default_frame], axis=0)))
        self.register_buffer('restype_atom14_to_rigid_group', torch.from_numpy(np.concatenate([rc.restype_atom14_to_rigid_group, rc.average_atom14_to_rigid_group], axis=0)))
        self.register_buffer('restype_atom14_mask', torch.from_numpy(np.concatenate([rc.restype_atom14_mask, rc.average_atom14_mask], axis=0)))
        self.register_buffer('restype_atom14_rigid_group_positions', torch.from_numpy(np.concatenate([rc.restype_atom14_rigid_group_positions, rc.average_atom14_rigid_group_positions], axis=0)))


    def _get_atom37_to_atom14(self, aatype):
        residx_atom37_to_atom14 = self.restype_atom37_to_atom14[aatype]
        residx_atom37_mask = self.restype_atom37_mask[aatype]

        return residx_atom37_to_atom14, residx_atom37_mask


    def get_atom37_to_atom14(self, aatype):
        """Get the redisue-wise atom indices for interconverting atom-37 to atom-14 representation.

        Args:
            aatype: [N_res]
                The residue type. 

        Returns:
            idx_atom37_to_atom14: [N_res, 37]
                The atom indices for interconverting atom-37 to atom-14 representation.
            
            mask_atom37_to_atom14: [N_res, 37]
                The mask for atom-37representation.
        """
        if isinstance(aatype, np.ndarray):
            residx_atom37_to_atom14, residx_atom37_mask = self._get_atom37_to_atom14(torch.from_numpy(aatype))
            residx_atom37_to_atom14 = residx_atom37_to_atom14.numpy()
            residx_atom37_mask = residx_atom37_mask.numpy()
        elif isinstance(aatype, torch.Tensor):
            residx_atom37_to_atom14, residx_atom37_mask = self._get_atom37_to_atom14(aatype)
            # residx_atom37_to_atom14 = torch.from_numpy(residx_atom37_to_atom14).to(aatype.device)
            # residx_atom37_mask = torch.from_numpy(residx_atom37_mask).to(aatype.device)
        
        return residx_atom37_to_atom14, residx_atom37_mask
    
    def _get_atom14_to_atom37(self, aatype):
        residx_atom14_to_atom37 = self.restype_atom14_to_atom37[aatype]
        residx_atom14_mask = self.restype_atom14_mask[aatype]

        return residx_atom14_to_atom37, residx_atom14_mask

    def get_atom14_to_atom37(self, aatype):
        """Get the redisue-wise atom indices for interconverting atom-14 to atom-37 representation.

        Args:
            aatype: [N_res]
                The residue type. 

        Returns:
            idx_atom14_to_atom37: [N_res, 14]
                The atom indices for interconverting atom-14 to atom-37 representation.
            
            mask_atom14_to_atom37: [N_res, 14]
                The mask for atom-14 representation.
        """
        if isinstance(aatype, np.ndarray):
            residx_atom14_to_atom37, residx_atom14_mask = self._get_atom14_to_atom37(torch.from_numpy(aatype))
            residx_atom14_to_atom37 = residx_atom14_to_atom37.cpu().numpy()
            residx_atom14_mask = residx_atom14_mask.cpu().numpy()
        elif isinstance(aatype, torch.Tensor):
            residx_atom14_to_atom37, residx_atom14_mask = self._get_atom14_to_atom37(aatype)
            # residx_atom14_to_atom37 = torch.from_numpy(residx_atom14_to_atom37).to(aatype.device)
            # residx_atom14_mask = torch.from_numpy(residx_atom14_mask).to(aatype.device)
        
        return residx_atom14_to_atom37, residx_atom14_mask
    

    def coord14_to_coord37(self, coord14: torch.Tensor, aa_type = None, coord14_mask = None, trunc = False,
                       default_res = 'G'):
        n_res, n_atom = coord14.shape[-3:-1]
        n_batch_dim = len(coord14.shape) - 2

        if aa_type is None:
            aa_type = torch.ones(n_res, dtype=torch.int64) * rc.restypes.index(default_res)

        if coord14_mask is None:
            coord14_mask = torch.ones(n_res, n_atom, dtype=torch.bool)
        
        #. [N_res, 14, 3]
        coord14 = torch.nn.functional.pad(coord14, (0, 0, 0, 14-n_atom), value=0)

        #. [N_res, 14]
        coord14_mask = torch.nn.functional.pad(coord14_mask.bool(), (0, 14-n_atom), value=0)

        #. [N_res, 37], [N_res, 37]
        coord37_to_coord14_idx, coord37_default_mask = self.get_atom37_to_atom14(aa_type)
        coord37_final_mask  = coord37_default_mask * torch.gather(coord14_mask, 1, coord37_to_coord14_idx)
        coord37_final_coord = coord37_final_mask[(None,) * (n_batch_dim-1)][..., None] * batched_gather(
                coord14,  #. [N_res, 14, 3]
                coord37_to_coord14_idx[(None,) * (n_batch_dim-1)],  #. [N_res, 37]
                dim=-2,
                no_batch_dims=n_batch_dim,
            )
        
        if trunc:
            last_idx = torch.where(coord37_final_mask.sum(dim=0))[0].max()
            coord37_final_mask = coord37_final_mask[..., :last_idx+1]
            coord37_final_coord = coord37_final_coord[..., :last_idx+1, :]

        return coord37_final_coord, coord37_final_mask

    
    def get_backbone_coord(self, backbone_frame: Rigid, 
                     aa_type: Optional[torch.Tensor] = None,
                     aa_mask: Optional[torch.Tensor] = None):
        """generate a `Protein` object with only backbone frame.

        Args:
            backbone_frame: [*, N_res]
                The backbone frame.
            res_idx: [*, N_res]
                The residue index.
            res_mask: [*, N_res]
                The residue mask.
        """
        if aa_mask is None:
            aa_mask = torch.ones(backbone_frame.shape, dtype=torch.bool, device=backbone_frame.device)
        elif isinstance(aa_mask, np.ndarray):
            aa_mask = torch.from_numpy(aa_mask).to(backbone_frame.device)
        
        if aa_type is None:
            aa_type = torch.ones(backbone_frame.shape, dtype=torch.long, device=backbone_frame.device) * 21
        elif isinstance(aa_type, np.ndarray):
            aa_type = torch.from_numpy(aa_type).to(backbone_frame.device)

        #. [*, N_res, 3]
        unsqueezed_frame = backbone_frame.unsqueeze(-1) * torch.ones((1, 3), device=backbone_frame.device)

        #. [*, N_res, 3, 3]
        lit_positions = self.restype_atom14_rigid_group_positions[:, :3][aa_type]

        #. [*, N_res, 3, 3]
        pred_positions = unsqueezed_frame.apply(lit_positions)
        pred_positions = pred_positions * aa_mask.unsqueeze(-1).unsqueeze(-1)

        return pred_positions

    def get_backbone_coord37(self, backbone_frame: Rigid, 
                     aa_type: Optional[torch.Tensor] = None,
                     aa_mask: Optional[torch.Tensor] = None,
                     return_mask: bool = False):
        if aa_mask is None:
            aa_mask = torch.ones(backbone_frame.shape, dtype=torch.bool)
        elif isinstance(aa_mask, np.ndarray):
            aa_mask = torch.from_numpy(aa_mask)
        
        if aa_type is None:
            aa_type = torch.ones(backbone_frame.shape, dtype=torch.long) * 21
        elif isinstance(aa_type, np.ndarray):
            aa_type = torch.from_numpy(aa_type)

        #. [*]
        batch_shape = backbone_frame.shape


        #. [*, N_res, 3, 3]
        pred_positions = self.get_backbone_coord(backbone_frame, aa_type, aa_mask)

        #. [*, N_res, 37, 3]
        pred_positions_37 = torch.zeros((*batch_shape, 37, 3), dtype=torch.float32, device=backbone_frame.device)

        # print(pred_positions_37.shape)
        # print(pred_positions.shape)
        pred_positions_37[..., :3, :] = pred_positions

        if return_mask:
            #. [*, N_res, 37]
            position_mask = torch.zeros((*batch_shape, 37), dtype=torch.long, device=backbone_frame.device)
            position_mask[..., :3] = 1
            position_mask = position_mask * aa_mask.unsqueeze(-1)
            return pred_positions_37, position_mask

        return pred_positions_37
    

    def get_backbone_protein(self, backbone_frame: Rigid, 
                     aa_type  = None,
                     aa_mask  = None,
                     residue_index: Optional[np.ndarray] = None,
                     b_factors: Optional[np.ndarray] = None,
                     chain_index: Optional[np.ndarray] = None,
                     remark: Optional[str] = None,
                     parents: Optional[Sequence[str]] = None,
                     parents_chain_index: Optional[Sequence[int]] = None):
        """protein dataclass with only backbone frame.
        """
        #. [*, N_res]
        if aa_mask is None:
            aa_mask = torch.ones(backbone_frame.shape, dtype=torch.bool, device=backbone_frame.device)
        elif isinstance(aa_mask, np.ndarray):
            aa_mask = torch.from_numpy(aa_mask).to(backbone_frame.device)
        
        #. [*, N_res]
        if aa_type is None:
            aa_type = torch.ones(backbone_frame.shape, dtype=torch.long, device=backbone_frame.device) * 21
        elif isinstance(aa_type, np.ndarray):
            aa_type = torch.from_numpy(aa_type).to(backbone_frame.device)

        #. [*, N_res, 3, 3]
        backbone_position = self.get_backbone_coord(backbone_frame, aa_type, aa_mask)
        #. [*, N_res]
        aa_type = torch.clamp(aa_type, 0, 20)

        prot = protein.from_nparray(aatype = aa_type.cpu().numpy(),
                                    atom_positions=backbone_position.cpu().numpy(),
                                    atom_mask=aa_mask.unsqueeze(-1).expand(-1, 3).cpu().numpy(),
                                    residue_index=residue_index,
                                    b_factors=b_factors,
                                    chain_index=chain_index,
                                    remark=remark,
                                    parents=parents,
                                    parents_chain_index=parents_chain_index)
        return prot
    

    def get_backbone_traj(self, backbone_frames: Rigid,
                          aa_type  = None,
                          aa_mask  = None,
                          residue_index: Optional[np.ndarray] = None,
                          b_factors: Optional[np.ndarray] = None,
                          chain_index: Optional[np.ndarray] = None,
                          remark: Optional[str] = None,
                          parents: Optional[Sequence[str]] = None,
                          parents_chain_index: Optional[Sequence[int]] = None,
                          label_override = None,
                          default_res = None
                          ):
        """list of protein dataclass with only backbone frame.

            backbone_frames: Rigid of size [N_traj, N_res] or list
            aa_type: [N_res]
            aa_mask: [N_res]
        """
        protein_list = []
        n_traj = len(backbone_frames)

        if default_res is None:
            default_res_idx = 21
        else:
            default_res_idx = rc.restypes.index(default_res)

        #. [*, N_res]
        if aa_mask is None:
            aa_mask = torch.ones(backbone_frames[0].shape, dtype=torch.bool, device=backbone_frames.device)
        elif isinstance(aa_mask, np.ndarray):
            aa_mask = torch.from_numpy(aa_mask).to(backbone_frames.device).bool()
        
        #. [*, N_res]
        if aa_type is None:
            aa_type = torch.ones(backbone_frames[0].shape, dtype=torch.long, device=backbone_frames.device) * default_res_idx
        elif isinstance(aa_type, np.ndarray):
            aa_type = torch.from_numpy(aa_type).to(backbone_frames.device).long()

        for i in range(n_traj):
            if residue_index is not None:
                cur_residue_index = residue_index if residue_index.ndim == 1 else residue_index[i]
            else:
                cur_residue_index = None
            if b_factors is not None:
                cur_b_factors = b_factors if b_factors.ndim == 1 else b_factors[i]
            else:
                cur_b_factors = None
            if chain_index is not None:
                cur_chain_index = chain_index if chain_index.ndim == 1 else chain_index[i]
            else:
                cur_chain_index = None
            if remark is not None:
                cur_remark = remark if isinstance(remark, str) else remark[i]
            else:
                cur_remark = None
            if parents is not None:
                cur_parents = parents if isinstance(parents, str) else parents[i]
            else:
                cur_parents = None
            if parents_chain_index is not None:
                cur_parents_chain_index = parents_chain_index if isinstance(parents_chain_index, int)  else parents_chain_index[i]
            else:
                cur_parents_chain_index = None
            if label_override is not None:
                cur_label_override = label_override if isinstance(label_override, str) else label_override[i]
            else:
                cur_label_override = None

            #. [*, N_res, 3, 3]
            backbone_position = self.get_backbone_coord(backbone_frames[i], aa_type, aa_mask)
            #. [*, N_res]
            # 21th type is only used for getting backbone average position, and is not valid for `protein` dataclass
            aa_type_clamped = torch.clamp(aa_type, 0, 20)
            cur_protein = protein.from_nparray(aatype = aa_type_clamped.cpu().numpy(),
                                    atom_positions=backbone_position.cpu().numpy(),
                                    atom_mask=aa_mask.unsqueeze(-1).expand(-1, 3).cpu().numpy(),
                                    residue_index=cur_residue_index,
                                    b_factors=cur_b_factors,
                                    chain_index=cur_chain_index,
                                    remark=cur_remark,
                                    parents=cur_parents,
                                    parents_chain_index=cur_parents_chain_index,
                                    label_override=cur_label_override
                                    )
            protein_list.append(cur_protein)
        
        return ProteinTraj(protein_list)
    

    def get_coord_traj(self, coord_frames: torch.Tensor,
                          aa_type  = None,
                          aa_mask  = None,
                          residue_index: Optional[np.ndarray] = None,
                          b_factors: Optional[np.ndarray] = None,
                          chain_index: Optional[np.ndarray] = None,
                          remark: Optional[str] = None,
                          parents: Optional[Sequence[str]] = None,
                          parents_chain_index: Optional[Sequence[int]] = None,
                          label_override = None,
                          default_res = None):
        """list of protein dataclass with only backbone frame.

            backbone_frames: Rigid of size [N_traj, N_res, N_atom, 3]
            aa_type: [N_res]
            aa_mask: [N_res, N_atom]
        """
        protein_list = []
        n_traj = coord_frames.shape[0]

        if default_res is None:
            default_res_idx = 21
        else:
            default_res_idx = rc.restypes.index(default_res)

        #. [*, N_res]
        if aa_mask is None:
            aa_mask = torch.ones(coord_frames[0].shape[:-1], dtype=torch.bool, device=coord_frames.device)
        elif isinstance(aa_mask, np.ndarray):
            aa_mask = torch.from_numpy(aa_mask).to(coord_frames.device).bool()
        
        #. [*, N_res]
        if aa_type is None:
            aa_type = torch.ones(coord_frames[0].shape[:-2], dtype=torch.long, device=coord_frames.device) * default_res_idx
        elif isinstance(aa_type, np.ndarray):
            aa_type = torch.from_numpy(aa_type).to(coord_frames.device).long()

        for i in range(n_traj):
            #. [N_res]
            if residue_index is not None:
                cur_residue_index = residue_index
            else:
                cur_residue_index = None

            #. [N_frame, N_res, N_atom]
            if b_factors is not None:
                cur_b_factors = b_factors if b_factors.ndim == 1 else b_factors[i]
            else:
                cur_b_factors = None
            
            #. [N_frame]
            if remark is not None:
                cur_remark = remark if isinstance(remark, str) else remark[i]
            else:
                cur_remark = None

            #. [N_frame]
            if parents is not None:
                cur_parents = parents if isinstance(parents, str) else parents[i]
            else:
                cur_parents = None

            #. [N_frame]
            if label_override is not None:
                cur_label_override = label_override if isinstance(label_override, str) else label_override[i]
            else:
                cur_label_override = None

            cur_chain_index = None
            cur_parents_chain_index = None

            #. [*, N_res, 3, 3]
            backbone_position = coord_frames[i]
            #. [*, N_res]
            # 21th type is only used for getting backbone average position, and is not valid for `protein` dataclass
            aa_type_clamped = torch.clamp(aa_type, 0, 20)
            cur_protein = protein.from_nparray(aatype = aa_type_clamped.cpu().numpy(),
                                    atom_positions=backbone_position.cpu().numpy(),
                                    atom_mask=aa_mask.cpu().numpy(),
                                    residue_index=cur_residue_index,
                                    b_factors=cur_b_factors,
                                    chain_index=cur_chain_index,
                                    remark=cur_remark,
                                    parents=cur_parents,
                                    parents_chain_index=cur_parents_chain_index,
                                    label_override=cur_label_override
                                    )
            protein_list.append(cur_protein)
        
        return ProteinTraj(protein_list)

        
    def get_backbone_protein37(self, backbone_frame: Rigid, 
                     aa_type  = None,
                     aa_mask  = None,
                     residue_index: Optional[np.ndarray] = None,
                     b_factors: Optional[np.ndarray] = None,
                     chain_index: Optional[np.ndarray] = None,
                     remark: Optional[str] = None,
                     parents: Optional[Sequence[str]] = None,
                     parents_chain_index: Optional[Sequence[int]] = None):
        #. [*, N_res]
        if aa_mask is None:
            aa_mask = torch.ones(backbone_frame.shape, dtype=torch.bool)
        elif isinstance(aa_mask, np.ndarray):
            aa_mask = torch.from_numpy(aa_mask)
        
        #. [*, N_res]
        if aa_type is None:
            aa_type = torch.ones(backbone_frame.shape, dtype=torch.long) * 21
        elif isinstance(aa_type, np.ndarray):
            aa_type = torch.from_numpy(aa_type)

        #. [*, N_res, 37, 3], [*, N_res, 37]
        backbone_position_37, backbone_mask_37 = self.get_backbone_coord37(backbone_frame, aa_type, aa_mask, return_mask=True)

        #. [*, N_res]
        aa_type = torch.clamp(aa_type, 0, 20)

        prot = protein.from_nparray(aatype = aa_type.cpu().numpy(),
                                    atom_positions=backbone_position_37.cpu().numpy(),
                                    atom_mask=backbone_mask_37.cpu().numpy(),
                                    residue_index=residue_index,
                                    b_factors=b_factors,
                                    chain_index=chain_index,
                                    remark=remark,
                                    parents=parents,
                                    parents_chain_index=parents_chain_index)
        return prot
    

class ProteinTraj():
    protein_list = []

    def __init__(self, protein_list = []):
        self.protein_list = protein_list

    
    def append(self, protein):
        self.protein_list.append(protein)

    def __getitem__(self, index):
        return self.protein_list[index]
    
    def __len__(self):
        return len(self.protein_list)
    
    def to_pdb(self) -> str:
        """Converts a `Protein` instance to a PDB string.
        Adapt from openfold.np.protein

        Args:
        prot: The protein to convert to PDB.

        Returns:
        PDB string.
        """
        #. 21 types of residue
        restypes = rc.restypes + ["X"]
        res_1to3 = lambda r: rc.restype_1to3.get(restypes[r], "UNK")

        #. 37 types of atom
        atom_types = rc.atom_types

        pdb_lines = []

        for pdb_idx, prot in enumerate(self.protein_list):

            atom_mask = prot.atom_mask
            aatype = prot.aatype
            atom_positions = prot.atom_positions
            residue_index = prot.residue_index.astype(np.int32)
            b_factors = prot.b_factors
            chain_index = prot.chain_index

            # atom_type_trunc_len = aatype.shape[0]
            # print(atom_type_trunc_len)

            if np.any(aatype > rc.restype_num):
                raise ValueError("Invalid aatypes.")

            headers = protein.get_pdb_headers(prot)
            if(len(headers) > 0):
                pdb_lines.extend(headers)

            n = aatype.shape[0]
            atom_index = 1
            prev_chain_index = 0
            chain_tags = string.ascii_uppercase
            # Add all atom sites.
            for i in range(n):
                res_name_3 = res_1to3(aatype[i])
                for atom_name, pos, mask, b_factor in zip(
                    atom_types[:], atom_positions[i, :], atom_mask[i, :], b_factors[i, :]
                ): # atom_type_trunc_len
                    if mask < 0.5:
                        continue

                    record_type = "ATOM"
                    name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                    alt_loc = ""
                    insertion_code = ""
                    occupancy = 1.00
                    element = atom_name[
                        0
                    ]  # Protein supports only C, N, O, S, this works.
                    charge = ""
            
                    chain_tag = "A"
                    if(chain_index is not None):
                        chain_tag = chain_tags[chain_index[i]]

                    # PDB is a columnar format, every space matters here!
                    atom_line = (
                        f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                        f"{res_name_3:>3} {chain_tag:>1}"
                        f"{residue_index[i]:>4}{insertion_code:>1}   "
                        f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                        f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                        f"{element:>2}{charge:>2}"
                    )
                    pdb_lines.append(atom_line)
                    atom_index += 1

                should_terminate = (i == n - 1)
                if(chain_index is not None):
                    if(i != n - 1 and chain_index[i + 1] != prev_chain_index):
                        should_terminate = True
                        prev_chain_index = chain_index[i + 1]

                if(should_terminate):
                    # Close the chain.
                    chain_end = "TER"
                    chain_termination_line = (
                        f"{chain_end:<6}{atom_index:>5}      "
                        f"{res_1to3(aatype[i]):>3} "
                        f"{chain_tag:>1}{residue_index[i]:>4}"
                    )
                    pdb_lines.append(chain_termination_line)
                    atom_index += 1

                    if(i != n - 1):
                        # "prev" is a misnomer here. This happens at the beginning of
                        # each new chain.
                        pdb_lines.extend(protein.get_pdb_headers(prot, prev_chain_index))

            pdb_lines.append("ENDMDL")
            pdb_lines.append("")

        pdb_lines.append("END")
        pdb_lines.append("")
        return "\n".join(pdb_lines)
    

    def save_pdb(self, filename, makeidirs=False, overwrite=False):
        dirname = os.path.dirname(filename)
        
        # check if dir exists
        if not os.path.exists(dirname):
            if makeidirs:
                os.makedirs(dirname)
            else:
                raise ValueError("Directory does not exist: {}".format(dirname))
        
        # check if file exists
        if os.path.exists(filename):
            if not overwrite:
                raise ValueError("File already exists: {}".format(filename))

        with open(filename, "w") as f:
            f.write(self.to_pdb())