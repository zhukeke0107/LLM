#!/usr/bin/env bash
#
# Sync this project between this machine and the remote dev box.
# Configure the target in .sync.conf; see --help for usage.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="$PROJECT_ROOT/.sync.conf"
IGNORE="$PROJECT_ROOT/.syncignore"

# .sync.conf sets the defaults; an env var of the same name wins.
if [ -f "$CONF" ]; then
  while IFS='=' read -r key value; do
    case "$key" in ''|\#*) continue ;; esac
    [ -n "${!key-}" ] || export "$key=$value"
  done < "$CONF"
fi

HOST="${SYNC_HOST-}"
REMOTE_PATH="${SYNC_REMOTE_PATH-}"

usage() {
  cat <<EOF
Usage: scripts/sync.sh <command> [options]

Commands:
  push          Copy local changes up to $HOST
  pull          Copy remote changes down to this machine
  status        Show what a push and a pull would each change (never transfers)
  check         Verify SSH reachability and rsync on both ends
  shell         Open an interactive shell in the remote project directory

Options:
  -n, --dry-run   List what would change without transferring
      --delete    Also remove files at the destination that no longer exist
                  at the source. Asks before running; pair with --dry-run first.
  -y, --yes       Skip the --delete confirmation prompt
  -q, --quiet     Only print the summary
  -h, --help      Show this message

Target: $HOST:$REMOTE_PATH
  (edit .sync.conf, or set SYNC_HOST / SYNC_REMOTE_PATH in the environment)

Excludes come from .syncignore. Large artifacts and .git/ are excluded by
default, so each machine keeps its own checkpoints and its own git state.
EOF
}

die() { printf 'sync: %s\n' "$1" >&2; exit 1; }

# rsync on macOS is openrsync, which rejects --info=progress2 and --mkpath;
# every flag used below works on both openrsync and GNU rsync 3.x.
RSYNC_FLAGS=(-a -z -h --itemize-changes --stats)
DRY_RUN=0
DELETE=0
ASSUME_YES=0
QUIET=0

CMD="${1-}"
[ $# -gt 0 ] && shift || true

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--dry-run) DRY_RUN=1 ;;
    --delete)     DELETE=1 ;;
    -y|--yes)     ASSUME_YES=1 ;;
    -q|--quiet)   QUIET=1 ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

case "$CMD" in
  -h|--help|'') usage; exit 0 ;;
esac

[ -n "$HOST" ]        || die "SYNC_HOST is not set (see .sync.conf)"
[ -n "$REMOTE_PATH" ] || die "SYNC_REMOTE_PATH is not set (see .sync.conf)"
case "$REMOTE_PATH" in /*) ;; *) die "SYNC_REMOTE_PATH must be absolute: $REMOTE_PATH" ;; esac

[ -f "$IGNORE" ] && RSYNC_FLAGS+=(--exclude-from "$IGNORE")
[ "$QUIET" -eq 1 ] || RSYNC_FLAGS+=(--progress)

ssh_run() { ssh -o ConnectTimeout=10 "$HOST" "$@"; }

# Confirm before a run that can delete files at the destination.
confirm_delete() {
  local dest="$1"
  [ "$DELETE" -eq 1 ] || return 0
  [ "$DRY_RUN" -eq 1 ] && return 0
  [ "$ASSUME_YES" -eq 1 ] && return 0
  printf '\n--delete will REMOVE files under %s that are absent at the source.\n' "$dest"
  printf 'Continue? [y/N] '
  local reply; read -r reply
  case "$reply" in [yY]|[yY][eE][sS]) ;; *) die "aborted" ;; esac
}

# Build the final flag list for one transfer.
transfer_flags() {
  local flags=("${RSYNC_FLAGS[@]}")
  [ "$DRY_RUN" -eq 1 ] && flags+=(--dry-run)
  [ "$DELETE" -eq 1 ]  && flags+=(--delete)
  printf '%s\n' "${flags[@]}"
}

do_push() {
  local flags=(); while IFS= read -r f; do flags+=("$f"); done < <(transfer_flags)
  confirm_delete "$HOST:$REMOTE_PATH"
  [ "$QUIET" -eq 1 ] || printf '==> push  %s/  ->  %s:%s/\n' "$PROJECT_ROOT" "$HOST" "$REMOTE_PATH"
  # openrsync has no --mkpath, so create the destination up front.
  ssh_run "mkdir -p '$REMOTE_PATH'"
  rsync "${flags[@]}" "$PROJECT_ROOT/" "$HOST:$REMOTE_PATH/"
}

do_pull() {
  local flags=(); while IFS= read -r f; do flags+=("$f"); done < <(transfer_flags)
  ssh_run "test -d '$REMOTE_PATH'" || die "$HOST:$REMOTE_PATH does not exist yet - push first"
  confirm_delete "$PROJECT_ROOT"
  [ "$QUIET" -eq 1 ] || printf '==> pull  %s:%s/  ->  %s/\n' "$HOST" "$REMOTE_PATH" "$PROJECT_ROOT"
  rsync "${flags[@]}" "$HOST:$REMOTE_PATH/" "$PROJECT_ROOT/"
}

case "$CMD" in
  push) do_push ;;
  pull) do_pull ;;

  status)
    DRY_RUN=1
    printf '===== push would change (local -> remote) =====\n'
    do_push || true
    printf '\n===== pull would change (remote -> local) =====\n'
    do_pull || true
    ;;

  check)
    printf 'local  rsync: %s\n' "$(rsync --version 2>&1 | head -1)"
    printf 'host        : %s\n' "$HOST"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
      'printf "remote      : %s (%s)\n" "$(hostname)" "$(whoami)"
       printf "remote rsync: %s\n" "$(rsync --version 2>&1 | head -1)"' \
      || die "cannot reach $HOST over SSH with key auth"
    if ssh_run "test -d '$REMOTE_PATH'"; then
      printf 'remote path : %s (exists)\n' "$REMOTE_PATH"
    else
      printf 'remote path : %s (not created yet - run: scripts/sync.sh push)\n' "$REMOTE_PATH"
    fi
    ;;

  shell)
    ssh_run "mkdir -p '$REMOTE_PATH'"
    exec ssh -t "$HOST" "cd '$REMOTE_PATH' && exec \${SHELL:-/bin/bash} -l"
    ;;

  *) die "unknown command: $CMD (try --help)" ;;
esac
