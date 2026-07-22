"""Same-box integration test for the --address/--token cluster join path
(mship_deploy.py's connect_ray join branch, modelship/deploy/serve_utils.py).

Token enforcement is a per-process gRPC interceptor keyed off env vars,
indifferent to whether the two Ray processes share a host — so a same-box test
is a legitimate proof that auth actually gates the join, no second VM required.
This is only possible because leave_ray_cluster tears down ONLY this process's
own in-process Ray node (ray._private.node.Node.kill_all_processes), scoped to
exactly the node the joiner started, never `ray stop` (which sweeps every Ray
process on the machine by name/cmdline, unscoped by cluster).

Deliberately its own file/process, separate from test_integration.py's shared
session-scoped `mship_cluster` fixture: this test starts its own independent
throwaway head (distinct GCS port + RAY_TMPDIR) and must never risk `ray stop`
or a stray kill touching that shared cluster.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time

import pytest

# Distinct from Ray's own 6379 default (test_integration.py's shared
# mship_cluster fixture starts a bare `ray start --head` on that default) and
# modelship's own-head default of 6380 — this is a throwaway, unrelated cluster.
_THROWAWAY_HEAD_PORT = 6480


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM the whole process group a `start_new_session=True` Popen created,
    escalating to SIGKILL on timeout. Used to tear down the throwaway HEAD (a
    real `ray start --head --block` subprocess) — never `ray stop`, which would
    sweep every Ray process on the machine."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def _poll(predicate, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        if predicate():
            return True
        time.sleep(1)
    return False


@pytest.fixture
def throwaway_head(tmp_path):
    """A fully independent Ray cluster — its own GCS port and RAY_TMPDIR — with
    token auth enabled, for exercising --address/--token same-box. Started
    supervised (--block, own process group) and torn down with a scoped
    SIGTERM, never `ray stop`.

    RAY_TMPDIR is a short-lived dir directly under /tmp (NOT pytest's tmp_path,
    which nests in the test's full name and blows past the 107-byte AF_UNIX
    socket path limit for Ray's plasma store socket).

    Yields (port, token, env).
    """
    head_home = tmp_path / "head_home"
    head_home.mkdir()
    head_ray_tmp = tempfile.mkdtemp(prefix="mship-join-test-head-")
    env = {**os.environ, "HOME": str(head_home), "RAY_TMPDIR": head_ray_tmp, "RAY_AUTH_MODE": "token"}

    # `ray start --head`'s CLI path never auto-generates a token (unlike
    # ray.init()'s own-cluster path) — it must already exist or startup fails.
    subprocess.run(["ray", "get-auth-token", "--generate"], env=env, check=True)
    token = (head_home / ".ray" / "auth_token").read_text().strip()

    proc = subprocess.Popen(
        [
            "ray",
            "start",
            "--head",
            f"--port={_THROWAWAY_HEAD_PORT}",
            "--include-dashboard=false",
            "--disable-usage-stats",
            "--block",
        ],
        env=env,
        start_new_session=True,
    )

    marker = os.path.join(head_ray_tmp, "ray", "ray_current_cluster")
    deadline = time.time() + 30
    try:
        while not os.path.exists(marker):
            if proc.poll() is not None:
                pytest.fail("Throwaway head process exited before starting up.")
            if time.time() > deadline:
                _terminate_process_group(proc)
                pytest.fail("Timed out waiting for the throwaway head to start.")
            time.sleep(0.5)

        try:
            yield _THROWAWAY_HEAD_PORT, token, env
        finally:
            _terminate_process_group(proc)
    finally:
        shutil.rmtree(head_ray_tmp, ignore_errors=True)


def _throwaway_head_node_count(env: dict) -> int:
    result = subprocess.run(
        ["ray", "status", f"--address=127.0.0.1:{_THROWAWAY_HEAD_PORT}"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.count("node_")


def _join_node_procs_alive(join_ray_tmp: str) -> bool:
    """True iff any of the joiner's own Ray node subprocesses (raylet, plasma
    store, agents — created in-process by serve_utils._join_ray_cluster via
    Node(head=False)) are still running. They're matched by the joiner's unique
    RAY_TMPDIR, which appears in their session/socket paths, so this is scoped to
    exactly this joiner and can't see the head's processes. Checking process
    existence directly — rather than polling `ray status`'s node count —
    sidesteps how long the *head's own* GCS takes to mark a killed node dead
    (observed, in this sandbox, to lag well past Ray's documented health-check
    defaults; a Ray-side convergence question, not something
    serve_utils.leave_ray_cluster's node teardown controls)."""
    result = subprocess.run(
        ["pgrep", "-f", join_ray_tmp],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


@pytest.mark.integration
@pytest.mark.cluster_join
class TestClusterJoin:
    def _joiner_env(self, tmp_path, suffix: str) -> tuple[dict, str]:
        """Returns (env, join_ray_tmp) for a joiner subprocess. join_ray_tmp is a
        short /tmp-rooted dir (see throwaway_head's docstring for why) that the
        caller must clean up."""
        join_home = tmp_path / suffix
        join_home.mkdir(exist_ok=True)
        join_ray_tmp = tempfile.mkdtemp(prefix=f"mship-join-test-{suffix}-")
        # PYTHONUNBUFFERED: the joiner's stdout is redirected to a log file (not a
        # TTY), so Python fully block-buffers it by default — without this, the
        # log can look truncated/stale for a long time even though the process is
        # actively progressing, which is misleading when diagnosing a hang.
        env = {**os.environ, "HOME": str(join_home), "RAY_TMPDIR": join_ray_tmp, "PYTHONUNBUFFERED": "1"}
        env.pop("RAY_AUTH_MODE", None)
        env.pop("RAY_AUTH_TOKEN", None)
        return env, join_ray_tmp

    def _run_joiner(self, tmp_path, head_port, token, suffix="join_home") -> subprocess.CompletedProcess:
        env, join_ray_tmp = self._joiner_env(tmp_path, suffix)
        args = [
            "uv",
            "run",
            "mship_deploy.py",
            "--address",
            f"127.0.0.1:{head_port}",
            "--node-num-cpus",
            "0",
            "--node-num-gpus",
            "0",
            "--no-metrics",  # avoid a real 8079 collision on this shared test box
            "--prune-ray-sessions",
            "false",
        ]
        if token is not None:
            args += ["--token", token]
        try:
            return subprocess.run(args, env=env, capture_output=True, text=True, timeout=90)
        finally:
            shutil.rmtree(join_ray_tmp, ignore_errors=True)

    def test_join_without_token_rejected(self, tmp_path, throwaway_head):
        head_port, _token, _env = throwaway_head
        result = self._run_joiner(tmp_path, head_port, token=None)
        assert result.returncode != 0, f"expected non-zero exit, got 0. stdout/err:\n{result.stdout}{result.stderr}"

    def test_join_with_wrong_token_rejected(self, tmp_path, throwaway_head):
        head_port, _token, _env = throwaway_head
        result = self._run_joiner(tmp_path, head_port, token="not-the-right-token")
        assert result.returncode != 0, f"expected non-zero exit, got 0. stdout/err:\n{result.stdout}{result.stderr}"

    def test_join_with_correct_token_adds_node_then_leaves_cleanly(self, tmp_path, throwaway_head):
        head_port, token, head_env = throwaway_head
        env, join_ray_tmp = self._joiner_env(tmp_path, "join_home")

        log_path = tmp_path / "joiner.log"
        try:
            with open(log_path, "w") as log_file:
                proc = subprocess.Popen(
                    [
                        "uv",
                        "run",
                        "mship_deploy.py",
                        "--address",
                        f"127.0.0.1:{head_port}",
                        "--token",
                        token,
                        "--gateway-name",
                        "join-test-gateway",
                        "--node-num-cpus",
                        "0",
                        "--node-num-gpus",
                        "0",
                        "--no-metrics",
                        "--prune-ray-sessions",
                        "false",
                    ],
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                try:
                    assert _poll(lambda: _throwaway_head_node_count(head_env) == 2, deadline_s=60), (
                        f"joiner did not appear as a second node within timeout. Log:\n{log_path.read_text()}"
                    )
                    assert proc.poll() is None, (
                        f"joiner exited early (code {proc.poll()}) instead of staying resident. "
                        f"Log:\n{log_path.read_text()}"
                    )

                    # Leave: SIGTERM the joiner (mirrors _cleanup's signal handling).
                    # _cleanup calls leave_ray_cluster, which tears down this node's
                    # in-process raylet/agents (Node.kill_all_processes) before the
                    # joiner process exits.
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=60)
                    # The raylet's own graceful-shutdown sequence (draining the
                    # gateway/coordinator actors this join hosted) runs independently
                    # and has been observed, in this sandbox, taking well past 15s even
                    # after the joiner process itself is gone — hence the generous
                    # window here rather than a tight one.
                    assert _poll(lambda: not _join_node_procs_alive(join_ray_tmp), deadline_s=60), (
                        f"joiner's Ray node subprocesses are still running after leave. Log:\n{log_path.read_text()}"
                    )
                    # The head itself must be unaffected — still answers `ray status` and
                    # was never sent a signal (leave is scoped to the joiner's own node,
                    # not `ray stop`/shutdown_ray(), either of which would take the
                    # throwaway head down too). Its OWN node count staying accurate here
                    # depends on Ray's own GCS convergence timing, not asserted on.
                    assert _throwaway_head_node_count(head_env) >= 1
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=10)
        finally:
            shutil.rmtree(join_ray_tmp, ignore_errors=True)
