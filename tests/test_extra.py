import os
import git

code_files = [".py", ".ipynb"]
text_files = [".md", ".qmd", ".txt"]
config_files = [".yml", ".toml", ".cfg"]
git_files = [".gitignore", ".gitmodules"]
other_files = [".template", ".bak"]


def check_file(file_name):
    allowed_extentions = code_files + text_files + config_files + git_files + other_files
    for ext in allowed_extentions:
        if file_name.endswith(ext):
            return True
    return False


def test_allowed():
    repo = git.Repo(search_parent_directories=True)
    all_files = repo.git.ls_files().split("\n")

    submodule_names = [submodule.name for submodule in repo.submodules]

    for file_name in all_files:
        if file_name in submodule_names:
            continue
        assert check_file(file_name), f"File {file_name} is not allowed"
