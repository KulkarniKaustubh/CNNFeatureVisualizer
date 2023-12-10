from setuptools import setup, find_packages

setup(
    name="torchboard",
    version="0.1.0",
    author="Kaustubh Milind Kulkarni, Jayant Duneja",
    author_email="kaustubhmilind.kulkarni@colorado.edu",
    packages=find_packages(),
    install_requires=[
        "torch",
        "matplotlib",
        "opencv-python",
        "pandas",
        "pillow",
    ],
)
