"""
Actor and Critic for DT2GS.

DT2GS generates a per-entity latent "subtask" instead of a "skill", but the
underlying mechanism (VAE context encoder producing a latent z with a KL +
reconstruction loss, optionally aggregated across entities and fed back into
the observation) is exactly what mcs's VAESkillGenerator + IntegrationComm /
IntegrationCritic already implement. main.py forces
args.skill_choice = "UseVAE" and args.num_skills = args.num_subtask for the
dt2gs algorithm, so we reuse the mcs networks directly here.
"""
from base_policy.algorithms.mcs.mcs_actor_critic import R_Actor, R_Critic

__all__ = ["R_Actor", "R_Critic"]
