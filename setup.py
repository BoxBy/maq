import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="maq",
    version="0.1.0",
    author="BoxBy",
    author_email="lute7071@gmail.com",
    description="Memory-Adaptive Quantization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BoxBy/Memory-Adaptive-Quantization",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
