from setuptools import setup, find_packages

setup(
    name="Sutton_RL_book",
    version="0.0.1",
    packages=find_packages(
        include=["envs", "envs.*"]
    ),  # Only include envs, not Chapter01
    install_requires=["gymnasium", "numpy"],
)
