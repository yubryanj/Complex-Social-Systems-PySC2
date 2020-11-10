from gym.envs.registration import register

register(
    id='collect_mineral_shards-v0',
    entry_point='collect_mineral_shard_env.envs:Collect_Mineral_Shard_Env',
)
