import setuptools
import subprocess


def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        readme = fh.read()
    return readme


def read_version():
    # this is not guaranteed to be a valid version string, but should work well enough
    git_describe = subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)

    if git_describe.returncode == 0:
        git_version = git_describe.stdout.strip().decode("utf-8")
    else:
        # per default no tags available in Github actions, but also no real need for a version
        git_version = "0.0.0"

    if git_version[0] == "v":
        git_version = git_version[1:]

    return git_version


setuptools.setup(
    name="abcvoting",
    version=read_version(),
    author="Martin Lackner",
    author_email="unexpected@sent.at",
    description="Python implementations of approval-based committee (multi-winner) rules",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/martinlackner/abcvoting/",
    project_urls={"Bug Tracker": "https://github.com/martinlackner/abcvoting/issues"},
    license="MIT License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    packages=["abcvoting"],
    python_requires=">=3.6",
    setup_requires=[
        "wheel",
    ],
    install_requires=[
        "networkx>=2.2",
        "ortools>=8.1.8487",
        "mip>=1.13.0",
        "ruamel.yaml >= 0.16.13",
    ],
    extras_require={
        "dev": [
            "pytest>=6",
            "coverage>=5.3",
            "black==20.8b1",
        ]
    },
)

# TODO Add optional requirements?
# TODO How about examples? Do we want to install them?
# TODO Add the test dir and test dependencies? Probably not?
