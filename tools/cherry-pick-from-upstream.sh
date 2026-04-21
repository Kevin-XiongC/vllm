#!/usr/bin/env bash
# Cherry-pick commits from a source branch onto a target branch, one at a time,
# for every commit strictly AFTER the given hash on the source branch.
#
# Defaults: source=upstream_main, target=main. Range is exclusive of <hash>,
# so pass the last commit you've already picked.
#
# Usage: tools/cherry-pick-from-upstream.sh <hash> [--source <branch>] [--target <branch>]

set -euo pipefail

SOURCE_BRANCH="upstream_main"
TARGET_BRANCH="main"
HASH=""

usage() {
    cat <<EOF
Usage: $0 <hash> [--source <branch>] [--target <branch>]

Cherry-picks every commit on <source> that comes after <hash> onto <target>,
one commit at a time, in chronological order. Uses 'git cherry-pick -x' so
each new commit records the original upstream hash.

Defaults:
  --source upstream_main
  --target main

<hash> is EXCLUSIVE: pass the last commit you have already picked.
On conflict, the script stops immediately so you can resolve manually.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE_BRANCH="$2"; shift 2 ;;
        --target) TARGET_BRANCH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)
            if [[ -z "$HASH" ]]; then
                HASH="$1"; shift
            else
                echo "error: unexpected argument: $1" >&2; usage >&2; exit 2
            fi
            ;;
    esac
done

if [[ -z "$HASH" ]]; then
    echo "error: missing starting commit hash" >&2
    usage >&2
    exit 2
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "error: not inside a git repository" >&2
    exit 2
fi

if ! git rev-parse --verify --quiet "${HASH}^{commit}" >/dev/null; then
    echo "error: '$HASH' is not a valid commit" >&2
    exit 2
fi

for b in "$SOURCE_BRANCH" "$TARGET_BRANCH"; do
    if ! git rev-parse --verify --quiet "refs/heads/$b" >/dev/null; then
        echo "error: local branch '$b' not found" >&2
        exit 2
    fi
done

if ! git merge-base --is-ancestor "$HASH" "$SOURCE_BRANCH"; then
    echo "error: $HASH is not an ancestor of $SOURCE_BRANCH" >&2
    exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "error: working tree is not clean; commit or stash first" >&2
    exit 1
fi

if [[ -e "$(git rev-parse --git-path CHERRY_PICK_HEAD)" ]]; then
    echo "error: a cherry-pick is already in progress. Resolve it first:" >&2
    echo "  git cherry-pick --continue   # or --abort / --skip" >&2
    exit 1
fi

# Oldest-first list of commits to pick (exclusive of HASH).
COMMITS=()
while IFS= read -r line; do
    COMMITS+=("$line")
done < <(git rev-list --reverse "${HASH}..${SOURCE_BRANCH}")

if [[ ${#COMMITS[@]} -eq 0 ]]; then
    echo "Nothing to pick: $SOURCE_BRANCH has no commits after $HASH."
    exit 0
fi

echo "Switching to $TARGET_BRANCH"
git checkout "$TARGET_BRANCH"

echo
echo "Will cherry-pick ${#COMMITS[@]} commit(s) from $SOURCE_BRANCH onto $TARGET_BRANCH:"
for c in "${COMMITS[@]}"; do
    echo "  $(git log -1 --format='%h %s' "$c")"
done
echo

for c in "${COMMITS[@]}"; do
    short=$(git rev-parse --short "$c")
    subject=$(git log -1 --format='%s' "$c")
    echo ">>> Cherry-picking $short $subject"
    if ! git cherry-pick -x "$c"; then
        echo
        echo "!!! Conflict on $short. To resume:"
        echo "  1. Resolve conflicts, 'git add' the files"
        echo "  2. git cherry-pick --continue   (or --abort)"
        echo "  3. Re-run this script to pick the remaining commits:"
        echo "       $0 $short --source $SOURCE_BRANCH --target $TARGET_BRANCH"
        exit 1
    fi
done

echo
echo "Done. Cherry-picked ${#COMMITS[@]} commit(s) onto $TARGET_BRANCH."
