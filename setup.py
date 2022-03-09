import os
import setuptools
import subprocess


def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        readme = fh.read()
    return readme


def read_version():
    """Read a version string.

    This is not guaranteed to be a valid version string, but should work well enough.
    Tested with version strings as tag names of the following formats:

    2.0.0
    2.0.0-beta
    v2.0.0
    v2.0.0-beta

    Version strings need to comply with PEP 440. Git tags are used, but for development versions
    the build number is appended. To comply with PEP 440 everything after the first dash is removed
    before appending the build number.
    """

    git_describe = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"], stdout=subprocess.PIPE
    )

    if git_describe.returncode == 0:
        git_version = git_describe.stdout.strip().decode("utf-8")
    else:
        # per default no old commit are fetched in Github actions, that means, that git describe
        # fails if the latest commit is not a tag, because old tags can't be found.
        # to fetch tags in Github actions use "fetch-depth: 0"
        raise RuntimeError("Git describe failed: did you fetch at least one tag?")

    # git_version contains the latest tag, if it is not identical with HEAD need to postifx
    head_is_tag = (
        subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"], stderr=subprocess.PIPE
        ).returncode
        == 0
    )
    if not head_is_tag:
        try:
            # set by Github actions, necessary for unique file names for PyPI
            build_nr = os.environ["BUILD_NUMBER"]
        except KeyError:
            build_nr = 0

        # +something is disallowed on PyPI even if allowed in PEP 440... :-/
        # https://github.com/pypa/pypi-legacy/issues/731#issuecomment-345461596

        next_stable = git_version.split("-")[0]
        git_version = f"{next_stable}.dev{build_nr}"

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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    packages=["abcvoting", "abcvoting.bipartite_matching"],
    python_requires=">=3.7",
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
            "coverage[toml]>=5.3",
            "black==22.1.0",
            "Sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
            "numpydoc>=1.2",
        ]
    },
)
