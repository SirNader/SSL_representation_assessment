import logging
import platform
import shlex
import sys
from pathlib import Path


def get_cli_command():
    # print the command with which the script was called
    # https://stackoverflow.com/questions/37658154/get-command-line-arguments-as-string
    script_str = f"python {Path(sys.argv[0]).name}"
    argstr = " ".join(map(shlex.quote, sys.argv[1:]))
    return f"{script_str} {argstr}"


def log_system_info():
    logging.info("------------------")
    logging.info("SYSTEM INFO")
    logging.info(f"host name: {platform.uname().node}")
    logging.info(f"OS: {platform.platform()}")
    logging.info(f"OS version: {platform.version()}")

    # print hash of latest git commit (git describe or similar stuff is a bit ugly because it would require the
    # git.exe path to be added in path as conda/python do something with the path and don't use the system
    # PATH variable by default)
    git_hash_file = Path(".git") / "FETCH_HEAD"
    if git_hash_file.exists():
        with open(git_hash_file) as f:
            lines = f.readlines()
            if len(lines) == 0:
                # this happened when I didn't have internet
                logging.warning(f".git/FETCH_HEAD has no content")
            else:
                git_hash = lines[0][:40]
                logging.info(f"current commit hash: {git_hash}")
    else:
        logging.warning("could not retrieve current git commit hash from ./.git/FETCH_HEAD")
