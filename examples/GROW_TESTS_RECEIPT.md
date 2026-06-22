# grow-tests live receipt (real Codestral FIM)

Subject (`test_clamp.py`, before):
```python
def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def test_clamp_within():
    assert clamp(5, 0, 10) == 5
```

Codestral FIM was seeded with `def test_` and returned 4 new tests (the raw
completion is committed at `examples/fixtures/grow_tests_codestral_middle.txt`).
Running it through the provably-additive gate with a REAL pytest subprocess:

```
[ADDITIVE] grow-tests
  P1 byte-identical prefix : PASS
  P2 parses                : PASS
  P3 node-ids monotonic    : PASS (+4 added, -0 removed)
  P4 green stays green     : PASS
  added: test_below, test_clamp_above, test_clamp_lo, test_clamp_hi
  NOTE: generated tests are CANDIDATES — review them for meaning.
```

- before node-ids: ['test_clamp_within']
- after node-ids : ['test_clamp_within', 'test_below', 'test_clamp_above', 'test_clamp_lo', 'test_clamp_hi']
- added          : ['test_below', 'test_clamp_above', 'test_clamp_lo', 'test_clamp_hi']
- every pre-existing test kept its outcome ({'test_clamp_within': 'passed'} -> {'test_clamp_within': 'passed', 'test_below': 'passed', 'test_clamp_above': 'passed', 'test_clamp_lo': 'passed', 'test_clamp_hi': 'passed'})

Replay offline (no key): `python examples/replay_grow_receipt.py`.
