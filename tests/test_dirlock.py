import os

from shutter.core.dirlock import DirLocks


def test_acquire_release_writes_owner_and_pid(tmp_path):
    a, b = tmp_path / "a.d", tmp_path / "sub" / "b.lock"
    locks = DirLocks([str(a), str(b)], owner="shutter-test")
    assert locks.try_acquire() is None
    assert locks.held
    for d in (a, b):
        assert (d / "owner").read_text().strip() == "shutter-test"
        assert (d / "pid").read_text().strip() == str(os.getpid())
    locks.release()
    assert not a.exists() and not b.exists()
    assert not locks.held


def test_all_or_nothing_and_reports_holder(tmp_path):
    a, b = tmp_path / "a.d", tmp_path / "b.lock"
    b.mkdir()
    (b / "owner_speed").write_text("longcat\n")
    (b / "pid").write_text("4242\n")
    locks = DirLocks([str(a), str(b)])
    reason = locks.try_acquire()
    assert reason is not None and "longcat" in reason and "4242" in reason
    assert not a.exists()          # earlier lock given back
    assert b.exists() and (b / "owner_speed").exists()  # theirs untouched
    assert not locks.held


def test_never_removes_foreign_or_stale_lock(tmp_path):
    a = tmp_path / "stale.d"
    a.mkdir()  # empty: looks stale
    locks = DirLocks(str(a))
    reason = locks.try_acquire()
    assert "stale" in reason
    locks.release()
    assert a.exists()
    assert locks.holders()[0]["path"] == str(a)


def test_dead_pid_is_reported_not_removed(tmp_path):
    a = tmp_path / "x.d"
    a.mkdir()
    (a / "owner").write_text("ghost\n")
    (a / "pid").write_text("999999999\n")
    reason = DirLocks(str(a)).try_acquire()
    assert "ghost" in reason and "stale" in reason
    assert a.exists()


def test_context_manager_and_second_instance_blocked(tmp_path):
    a = tmp_path / "a.d"
    with DirLocks(str(a), poll_s=0.01) as l1:
        assert a.exists()
        other = DirLocks(str(a), owner="other")
        assert "shutter" in other.try_acquire()
        other.release()
        assert a.exists()
    assert not a.exists()


def test_empty_paths_is_noop(tmp_path):
    locks = DirLocks([])
    assert locks.try_acquire() is None
    locks.release()
