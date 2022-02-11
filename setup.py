from setuptools import find_packages, setup


setup(
    name="hf_search",
    version="0.2",
    package_dir={"": "src"},
    packages=find_packages("src"),
    url="https://github.com/nouamanetazi/hf_search",
    license="MIT",
    author="nouamanetazi",
    author_email="nouamane98@gmail.com",
    description="hf semantic search",
    install_requires=[
        "huggingface_hub==0.4.0",
        "numpy",
        "pandas",
        "rank_bm25",
        "scikit_learn",
        "sentence_transformers",
        "setuptools",
        "tqdm",
    ],
)
