from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "pytorchtextvae",
    version = "0.0.1",
    description = "A partial reimplementation of \"Generating Sentences From a Continuous Space\" by Bowman, Vilnis, Vinyals, Dai, Jozefowicz, Bengio (https://arxiv.org/abs/1511.06349).",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    license = "MIT",
    url = "https://github.com/iconix/pytorch-text-vae",
    packages = [ 'pytorchtextvae' ],
    install_requires = [ 'dill', 'fire', 'pytorch', 'unidecode' ],
    keywords = [ 'deeplearning', 'pytorch', 'vae', 'nlp' ],
    classifiers = ['Development Status :: 3 - Alpha',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.6',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
)
