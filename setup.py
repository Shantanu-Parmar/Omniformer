from setuptools import setup, find_packages

setup(
    name="omniformer",
    version="0.1.1",
    description="Omniformer: Context-aware Transformer for GW Signal Classification",
    author="Shantanu Parmar",
    packages=find_packages(),
    package_data={"omniformer": ["config.py"]},
    install_requires=[
        "torch>=1.13",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "streamlit",
        "fastapi",
        "uvicorn"
    ],
    entry_points={
        "console_scripts": [
            "omniformer-train=omniformer.train:main",
            "omniformer-infer=omniformer.inference:main",
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
)
