from setuptools import setup

setup(name='collect_mineral_shard_env',
    version='0.0.1',
    install_requires=[
        'absl-py==0.11.0',
        'baselines @ git+https://github.com/openai/baselines@ea25b9e8b234e6ee1bca43083f8f3cf974143998',
        'cloudpickle==1.6.0',
        'gym==0.15.7',
        'numpy==1.18.5',
        'pandas==1.1.4',
        'Pillow==8.0.1',
        'pygame==2.0.0',
        'pyglet==1.5.0',
        'PySC2==3.0.0',
        's2clientprotocol==5.0.2.81102.0',
        's2protocol==5.0.3.81433.0',
        'stable-baselines @ git+https://github.com/hill-a/stable-baselines@b3f414f4f2900403107357a2206f80868af16da3',
        'tensorboard==1.15.0',
        'tensorflow==1.15.0',
        'tensorflow-estimator==1.15.1'
    ]
)
