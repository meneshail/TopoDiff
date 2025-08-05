from setuptools import setup, find_packages


setup(
    name="TopoDiff",
    version="1.1",
    packages=find_packages(),
    install_requires=[],

    entry_points={
        'console_scripts': [
            'topodiff-sample = run_sampling:main',
            'topodiff-train = run_training:main',
            'topodiff-preprocess = TopoDiff.script.preprocess_feat:main',
        ]
    },
)
