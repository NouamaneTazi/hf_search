from setuptools import find_packages, setup


setup(
    name='hf_search',
    version='0.1',
    package_dir={"": "src"},
    packages=find_packages("src"),
    url='https://github.com/nouamanetazi/hf_search',
    license='mit',
    author='nouamanetazi',
    author_email='nouamane98@gmail.com',
    description='hf semantic search'
)