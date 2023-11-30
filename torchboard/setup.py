from setuptools import setup, find_packages

setup(
    name="torchboard",
    version="0.1.0",
    author="Kaustubh Milind Kulkarni, Jayant Duneja",
    author_email="kaustubhmilind.kulkarni@colorado.edu",
    packages=find_packages(),
    install_requires=[
        "torch==2.1.0",
        "matplotlib==3.8.0",
        "opencv-python==4.8.1.78",
        "pandas==2.1.2",
        "pillow==10.0.1",
    ],
)
