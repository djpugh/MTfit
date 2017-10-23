"""release.py
*************
Build and tag a release, merging if --merge is set as a flag
"""
import subprocess
import sys
import os
import argparse

from git import Repo, InvalidGitRepositoryError

from mtfit import __version__


class DirtyRepository(Exception):
    pass


class BuildTestsFailed(Exception):
    pass


def build_release():
    """Build a release for the current repo"""
    # Check if there is a setup.py
    try:
        os.path.chdir(os.path.dirname(os.path.dirname(os.getcwd())))
        try:
            repo = Repo('.')
        except InvalidGitRepositoryError:
            pass
        if repo.is_dirty(untracked_files=True):
            raise DirtyRepository('The repository should not have any unstaged, untracked or uncommited files')
        # Try to run tests and check result
        result = subprocess.call([sys.executable, 'setup.py', 'test'])
        if result:
            raise BuildTestsFailed('Tests failed, cancelling build')
        # Try to build docs
        proc = subprocess.Popen([sys.executable, 'setup.py', 'build_docs'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        std_out, std_err = proc.communicate()
        sys.stdout.write(std_out)
        sys.stderr.write(std_err)
        try:
            proc.kill()
        except:
            pass
        add_and_commit(repo, "Documentation build")
        # Check if merging (git flow or otherwise?) - merge into master and then tag master and then merge master into develop
        options = _get_args()
        if options['merge']:
            merge_to_root(repo, options['root_branch'])
        # else tag commit
        repo.create_tag(options['release_tag'])
        # merge into develop if it exists
        if options['merge']:
            if 'develop' in repo.heads:
                merge_to_root(repo, 'develop')
            elif 'dev' in repo.heads:
                merge_to_root(repo, 'dev')
            repo.heads[options['root_branch']].checkout()
        # In tagged commit, build all sdists
        subprocess.call([sys.executable, 'setup.py', 'build_all'])
    except (ValueError, DirtyRepository, BuildTestsFailed) as e:
        print('\nError: '+e.message+'\n')
        sys.exit(1)
    except Exception as e:
        import traceback
        print('\n')
        traceback.print_exc()
        sys.exit(1)


def _get_args(args=None):
    """Build parser and parse args"""
    # build parser
    parser = argparse.ArgumentParser(prog='python_build_release', description="Automatically build a python release with tests")
    parser.add_argument('release_tag', type=str, help="Tag for the release", nargs=1)
    parser.add_argument('--merge', '-m', action='store_true', default=False, dest='merge')
    parser.add_argument('--version', '-v', action='version', version='%(prog)s'+__version__)
    parser.add_argument('--root-branch', '-r', type=str, default='master', help="Default branch to build a merged release on", dest="root_branch")
    if args is not None:
        options = parser.parse_args(args)
    else:
        options = parser.parse_args()
    return vars(options)


def add_and_commit(repo, message):
    """Add all files and commit to the repo"""
    # Check that files exist
    add_files = repo.untracked_files + [diff.a_path for diff in repo.index.diff(None)] + [diff.a_path for diff in repo.index.diff("HEAD")]
    if len(add_files):
        # Add untracked files
        repo.index.add(repo.untracked_files)
        # Unstaged files
        repo.index.add([diff.a_path for diff in repo.index.diff(None)])
        # Untracked files
        repo.index.add([diff.a_path for diff in repo.index.diff(None)])
        # Commit
        repo.index.commit(message)


def merge_to_root(repo, root_branch):
    """Merge the current branch to the root branch"""
    release_branch = repo.active_branch
    root_branch = repo.heads[root_branch].checkout()
    merge_base = repo.merge_base(root_branch, release_branch)
    repo.index.merge_tree(release_branch, base=merge_base)
    repo.index.commit("Merged "+release_branch.name+" into "+root_branch.name,
                      parent_commits=(root_branch.commit, release_branch.commit))
    repo.head.reset(index=True, working_tree=True)


if __name__ == "__main__":
    build_release()
