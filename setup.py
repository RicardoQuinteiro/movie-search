from setuptools import setup, find_packages

setup(
    name="movie_search",
    version="0.1",
    description="Library to implement AI-based search for movies",
    author="Ricardo Quinteiro",
    packages=find_packages(),
    install_requires=[
        "transformers==4.46.3",
        "haystack-ai==2.5.0",
        "sentence-transformers==3.0.1"
    ]
)