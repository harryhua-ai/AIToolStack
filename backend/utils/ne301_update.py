"""
NE301 project auto-update module
Automatically check and update NE301 project to latest version on startup
"""
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from backend.config import settings

logger = logging.getLogger(__name__)

NE301_REPO_URL = "https://github.com/camthink-ai/ne301.git"
DEFAULT_NE301_PATH = Path("/app/ne301")


@dataclass
class UpdateResult:
    """Result of an update operation"""
    success: bool
    action: str  # "updated", "already_latest", "skipped", "failed"
    old_commit: str = ""
    new_commit: str = ""
    message: str = ""
    error: str = ""


def run_git_command(
    ne301_path: Path,
    args: List[str],
    timeout: int = 30,
    capture_output: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a git command in the NE301 directory

    Args:
        ne301_path: Path to NE301 project
        args: Git command arguments
        timeout: Command timeout in seconds
        capture_output: Whether to capture stdout/stderr

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = ["git"] + args
    logger.debug(f"Running git command: {' '.join(cmd)}")

    return subprocess.run(
        cmd,
        cwd=ne301_path,
        capture_output=capture_output,
        text=True,
        timeout=timeout
    )


def check_update_available(
    ne301_path: Path,
    timeout: int = 10
) -> Tuple[bool, str, str]:
    """
    Check if there's an update available for NE301

    Args:
        ne301_path: Path to NE301 project
        timeout: Operation timeout in seconds

    Returns:
        Tuple of (has_update, local_commit, remote_commit)
    """
    try:
        # Get current commit
        result = run_git_command(ne301_path, ["rev-parse", "HEAD"], timeout=timeout)
        if result.returncode != 0:
            logger.warning(f"Failed to get local commit: {result.stderr}")
            return False, "", ""

        local_commit = result.stdout.strip()

        # Fetch remote updates
        result = run_git_command(ne301_path, ["fetch", "origin"], timeout=timeout)
        if result.returncode != 0:
            logger.warning(f"Failed to fetch remote: {result.stderr}")
            return False, local_commit, ""

        # Get remote commit (assuming main branch)
        result = run_git_command(
            ne301_path,
            ["rev-parse", "origin/main"],
            timeout=timeout
        )
        if result.returncode != 0:
            # Try 'master' branch as fallback
            result = run_git_command(
                ne301_path,
                ["rev-parse", "origin/master"],
                timeout=timeout
            )
            if result.returncode != 0:
                logger.warning(f"Failed to get remote commit: {result.stderr}")
                return False, local_commit, ""

        remote_commit = result.stdout.strip()

        # Compare commits
        has_update = local_commit != remote_commit

        return has_update, local_commit, remote_commit

    except subprocess.TimeoutExpired:
        logger.warning(f"Git command timed out checking for updates")
        return False, "", ""
    except Exception as e:
        logger.warning(f"Error checking for updates: {e}")
        return False, "", ""


def check_local_changes(ne301_path: Path) -> Dict[str, any]:
    """
    Check for local uncommitted changes in NE301

    Args:
        ne301_path: Path to NE301 project

    Returns:
        Dictionary with change status:
        {
            "has_changes": bool,
            "staged": List[str],
            "unstaged": List[str],
            "untracked": List[str]
        }
    """
    result = {
        "has_changes": False,
        "staged": [],
        "unstaged": [],
        "untracked": []
    }

    try:
        # Check git status --porcelain
        git_result = run_git_command(
            ne301_path,
            ["status", "--porcelain"],
            timeout=10
        )

        if git_result.returncode != 0:
            logger.warning(f"Failed to check git status: {git_result.stderr}")
            return result

        # Parse output
        for line in git_result.stdout.strip().split('\n'):
            if not line:
                continue

            status = line[:2]
            filepath = line[3:]

            # Staged changes (first char is not space or ?)
            if status[0] in 'MADRC':
                result["staged"].append(filepath)

            # Unstaged changes (second char is not space)
            if status[1] == 'M':
                result["unstaged"].append(filepath)

            # Untracked files
            if status[0] == '?':
                result["untracked"].append(filepath)

        result["has_changes"] = bool(
            result["staged"] or result["unstaged"] or result["untracked"]
        )

    except Exception as e:
        logger.warning(f"Error checking local changes: {e}")

    return result


def update_ne301(
    ne301_path: Path,
    stash_changes: bool = False,
    timeout: int = 30
) -> UpdateResult:
    """
    Update NE301 project to latest version

    Args:
        ne301_path: Path to NE301 project
        stash_changes: Whether to stash local changes before updating
        timeout: Operation timeout in seconds

    Returns:
        UpdateResult with update status
    """
    # Get current commit
    try:
        result = run_git_command(ne301_path, ["rev-parse", "HEAD"], timeout=timeout)
        if result.returncode != 0:
            return UpdateResult(
                success=False,
                action="failed",
                error=f"Failed to get current commit: {result.stderr}"
            )
        old_commit = result.stdout.strip()
    except Exception as e:
        return UpdateResult(
            success=False,
            action="failed",
            error=f"Failed to get current commit: {e}"
        )

    # Check for local changes
    local_changes = check_local_changes(ne301_path)

    if local_changes["has_changes"]:
        if stash_changes:
            # Stash local changes
            try:
                import time
                stash_message = f"Auto-stash before update: {time.strftime('%Y%m%d_%H%M%S')}"
                result = run_git_command(
                    ne301_path,
                    ["stash", "push", "-m", stash_message],
                    timeout=timeout
                )
                if result.returncode != 0:
                    return UpdateResult(
                        success=False,
                        action="failed",
                        old_commit=old_commit,
                        error=f"Failed to stash changes: {result.stderr}"
                    )
                logger.info(f"Stashed local changes: {stash_message}")
            except Exception as e:
                return UpdateResult(
                    success=False,
                    action="failed",
                    old_commit=old_commit,
                    error=f"Failed to stash changes: {e}"
                )
        else:
            # Don't update, skip with message
            return UpdateResult(
                success=True,
                action="skipped",
                old_commit=old_commit,
                message="Local changes detected. Set NE301_UPDATE_STASH_CHANGES=True to auto-stash."
            )

    # Perform update
    try:
        # Ensure we're on main branch
        run_git_command(ne301_path, ["checkout", "main"], timeout=timeout)

        # Pull latest changes
        result = run_git_command(
            ne301_path,
            ["pull", "origin", "main"],
            timeout=timeout
        )

        if result.returncode != 0:
            return UpdateResult(
                success=False,
                action="failed",
                old_commit=old_commit,
                error=f"Failed to pull updates: {result.stderr}"
            )

        # Get new commit
        result = run_git_command(ne301_path, ["rev-parse", "HEAD"], timeout=timeout)
        if result.returncode != 0:
            return UpdateResult(
                success=True,
                action="updated",
                old_commit=old_commit,
                message="Update completed but could not verify new commit"
            )

        new_commit = result.stdout.strip()

        # Check if actually updated
        if new_commit == old_commit:
            return UpdateResult(
                success=True,
                action="already_latest",
                old_commit=old_commit,
                new_commit=new_commit
            )

        return UpdateResult(
            success=True,
            action="updated",
            old_commit=old_commit,
            new_commit=new_commit,
            message=f"Successfully updated from {old_commit[:8]} to {new_commit[:8]}"
        )

    except subprocess.TimeoutExpired:
        return UpdateResult(
            success=False,
            action="failed",
            old_commit=old_commit,
            error="Update operation timed out"
        )
    except Exception as e:
        return UpdateResult(
            success=False,
            action="failed",
            old_commit=old_commit,
            error=f"Update failed: {e}"
        )


def ensure_ne301_updated(
    ne301_path: Optional[Path] = None,
    timeout: int = 30,
    auto_update: Optional[bool] = None,
    stash_changes: Optional[bool] = None
) -> Tuple[Path, UpdateResult]:
    """
    Ensure NE301 project is at the latest version (main entry point)

    Args:
        ne301_path: Path to NE301 project (if None, will use default or detect)
        timeout: Operation timeout in seconds
        auto_update: Whether to auto-update (if None, use config)
        stash_changes: Whether to stash local changes (if None, use config)

    Returns:
        Tuple of (ne301_path, UpdateResult)
    """
    # Import ensure_ne301_project to make sure project exists
    from backend.utils.ne301_init import ensure_ne301_project

    # Step 1: Ensure project exists
    if ne301_path is None:
        ne301_path = ensure_ne301_project()
    else:
        ne301_path = Path(ne301_path).resolve()

    # Check if directory is a git repository
    git_dir = ne301_path / ".git"
    if not git_dir.exists():
        logger.warning(f"Not a git repository: {ne301_path}")
        return ne301_path, UpdateResult(
            success=False,
            action="skipped",
            message="Not a git repository, cannot check for updates"
        )

    # Step 2: Check for updates
    has_update, local_commit, remote_commit = check_update_available(
        ne301_path, timeout=timeout
    )

    if not has_update:
        if local_commit and remote_commit:
            logger.info(f"NE301 is already up to date: {local_commit[:8]}")
        else:
            logger.info("Could not verify update status, assuming up to date")

        return ne301_path, UpdateResult(
            success=True,
            action="already_latest",
            old_commit=local_commit,
            new_commit=local_commit,
            message="Already at latest version"
        )

    # Step 3: Decide whether to update
    if auto_update is None:
        auto_update = getattr(settings, 'NE301_AUTO_UPDATE', True)

    if not auto_update:
        logger.info(f"NE301 update available but auto-update is disabled")
        return ne301_path, UpdateResult(
            success=True,
            action="skipped",
            old_commit=local_commit,
            new_commit=remote_commit,
            message="Update available but auto-update is disabled"
        )

    # Step 4: Perform update
    if stash_changes is None:
        stash_changes = getattr(settings, 'NE301_UPDATE_STASH_CHANGES', False)

    update_result = update_ne301(
        ne301_path,
        stash_changes=stash_changes,
        timeout=timeout
    )

    return ne301_path, update_result


if __name__ == "__main__":
    # Test the module
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    # Test with default path
    path, result = ensure_ne301_updated()

    print(f"\nNE301 path: {path}")
    print(f"Result: {result.action}")
    if result.success:
        print(f"Message: {result.message}")
        if result.old_commit and result.new_commit:
            print(f"Commit: {result.old_commit[:8]} -> {result.new_commit[:8]}")
    else:
        print(f"Error: {result.error}")

    sys.exit(0 if result.success else 1)
