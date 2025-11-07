"""
Tracking module
---------------
- Tracking ligero para asignar IDs estables
"""
from model.tracking.simple_tracker import assign_stable_ids, calculate_object_similarity

__all__ = ['assign_stable_ids', 'calculate_object_similarity']
