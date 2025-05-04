from setuptools import setup

setup(
    name="topodiff_eval",
    packages=[
        'openfold',
        'topodiff_eval'
    ],
    package_dir={
        'openfold': './openfold',
        'topodiff_eval': './topodiff_eval'
    },
)
