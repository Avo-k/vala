"""Pool-aware time management — regression for the Yj41D0Md time-forfeit.

The deployed bot (Pool=2) flagged a 5+3 game because timectl sized the deep
search for a big pool: a ~6s budget launched a depth-3/top5 search that takes
~60s on 2 workers. timectl must now refuse searches that don't fit the pool.
"""
from __future__ import annotations

from vala.timectl import plan

# aggressive profile knobs (the deployed profile)
AGG_DEPTH, AGG_TR = 3, 5


def test_small_pool_blitz_does_not_run_deep():
    # 5+3, early move → ~6s budget, Pool=2. Must NOT launch the deep search, but
    # the cheap depth-1 screen-bait should still fit (so vala can trap in blitz).
    pl = plan(6000, AGG_DEPTH, AGG_TR, pool_size=2)
    assert not pl.run_deep, "Pool=2 blitz cannot afford the deep search"
    assert pl.run_screen, "Pool=2 blitz should still afford the depth-1 screen-bait"
    assert pl.human_depth == 1


def test_too_little_time_plays_fast_best():
    # Tiny budget on a small pool: not even the screen fits → engine-best.
    pl = plan(700, AGG_DEPTH, AGG_TR, pool_size=2)
    assert not pl.run_deep and not pl.run_screen


def test_small_pool_never_picks_depth3():
    # Even with a big budget, Pool=2 should stay shallow/narrow if it runs deep at all.
    pl = plan(12000, AGG_DEPTH, AGG_TR, pool_size=2)
    if pl.run_deep:
        assert pl.human_depth <= 2 and pl.top_replies <= 3


def test_big_pool_blitz_runs_deep_capped_at_2():
    pl = plan(6000, AGG_DEPTH, AGG_TR, pool_size=16)
    assert pl.run_deep
    assert pl.human_depth == 2, "blitz budget must cap depth at 2 even for aggressive"


def test_depth3_only_with_real_time():
    # Long budget + big pool may use depth 3; short budget must not.
    deep = plan(12000, AGG_DEPTH, AGG_TR, pool_size=16)
    assert deep.run_deep and deep.human_depth == 3
    shallow = plan(3000, AGG_DEPTH, AGG_TR, pool_size=16)
    assert shallow.human_depth <= 2


def test_bullet_plays_fast_best():
    pl = plan(300, AGG_DEPTH, AGG_TR, pool_size=16)
    assert not pl.run_deep


def test_richer_search_fits_a_bigger_pool():
    # More workers → can afford a richer search at the same budget.
    small = plan(6000, AGG_DEPTH, AGG_TR, pool_size=4)
    big = plan(6000, AGG_DEPTH, AGG_TR, pool_size=16)
    # big should run deep with at least as many replies as small (or small fast-best)
    assert big.run_deep
    if small.run_deep:
        assert big.top_replies >= small.top_replies
