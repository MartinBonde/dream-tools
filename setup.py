import setuptools

with open("README.md", "r") as file:
  long_description = file.read()

setuptools.setup(
  name="dream-tools",
  version="0.4.8",
  author="Martin Kirk Bonde",
  author_email="martin@bonde.dk",
  description="A collection of tools used by the Danish institute for economic modelling and forecasting, DREAM (http://dreammodel.dk).",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/MartinBonde/dream-tools",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
  ],
  python_requires='>=3.6',
  install_requires=["pandas", "easygui"],
)