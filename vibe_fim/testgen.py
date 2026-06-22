"""Provably-additive test generation via Codestral FIM.

Append new ``test_*`` functions to a test file and *prove* the change is additive:

  P1  byte-identical prefix — every pre-existing test is the same source, not
      regenerated-and-hopefully-equivalent (the original file is prepended verbatim);
  P2  parse-gate — the assembled file still parses;
  P3  node-id monotonicity — the run collects every pre-existing test id plus the
      new ones, and nothing was renamed/removed/shadowed;
  P4  green-stays-green by EXECUTION — each pre-existing test keeps the outcome it
      had on the original file, and each new test was actually collected and run.

``additive`` is the AND of all four. If it fails, nothing is written. The run is
the proof, so the pytest runner is injectable (unit-testable with no subprocess),
exactly like the FIM backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .fim import _check_parses, _trim_boundary_overlap, codestral_fim

# A test runner: file_text -> {node_id: outcome}, outcome in
# {"passed","failed","error","skipped"}. Injectable for tests.
Runner = Callable[[str], dict]

# We SEED the FIM prefix with the start of a test function. An empty region
# between bare sentinels makes Codestral return nothing (it thinks the file is
# done); seeding `def test_` makes it actually write tests. The byte-identity
# guarantee (P1) is independent of this — the original file is prepended verbatim.
_SEED = "def test_"


def _strip_fences(s: str) -> str:
    return "\n".join(ln for ln in s.splitlines() if not ln.lstrip().startswith("```"))


@dataclass(frozen=True)
class GrowTestsResult:
    text: str                 # the new file (original + generated tests), or original on failure
    generated: str            # the inserted test source
    before_ids: tuple
    after_ids: tuple
    outcomes_before: dict = field(default_factory=dict)
    outcomes_after: dict = field(default_factory=dict)
    prefix_identical: bool = False   # P1
    parses: bool | None = None       # P2
    ids_monotonic: bool = False      # P3
    green_preserved: bool = False    # P4
    ran: bool = False                # whether a runner was used (P3/P4 evaluated)

    @property
    def added_ids(self) -> tuple:
        return tuple(i for i in self.after_ids if i not in set(self.before_ids))

    @property
    def removed_ids(self) -> tuple:
        return tuple(i for i in self.before_ids if i not in set(self.after_ids))

    @property
    def additive(self) -> bool:
        structural = self.prefix_identical and self.parses is True
        if not self.ran:
            return False  # without a run, "additive" is not proven — only inserted
        return structural and self.ids_monotonic and self.green_preserved

    def report(self) -> str:
        def mark(b):
            return "PASS" if b else "FAIL"
        lines = [
            f"[{'ADDITIVE' if self.additive else 'NOT ADDITIVE'}] grow-tests",
            f"  P1 byte-identical prefix : {mark(self.prefix_identical)}",
            f"  P2 parses                : {mark(self.parses is True)}",
        ]
        if self.ran:
            lines += [
                f"  P3 node-ids monotonic    : {mark(self.ids_monotonic)} "
                f"(+{len(self.added_ids)} added, -{len(self.removed_ids)} removed)",
                f"  P4 green stays green     : {mark(self.green_preserved)}",
            ]
        else:
            lines.append("  P3/P4 not evaluated (no runner; pass one to prove additive)")
        if self.added_ids:
            lines.append("  added: " + ", ".join(self.added_ids))
        lines.append("  NOTE: generated tests are CANDIDATES — review them for meaning.")
        return "\n".join(lines)


def grow_tests(
    text: str,
    *,
    filename: str = "tests.py",
    focus: str = "",
    fim_fn: Callable | None = None,
    runner: Runner | None = None,
) -> GrowTestsResult:
    """Generate additive tests for `text` (a pytest file). Returns a result whose
    `.additive` is True only if P1-P4 hold; on failure `.text` is the original."""
    fim_fn = fim_fn or codestral_fim

    # Build the FIM left context: the whole file, a header comment, and the seed.
    body = text if text.endswith("\n") else text + "\n"
    header = "# Generated tests (candidates — review for meaning)"
    if focus:
        header += f": {focus}"
    lead = body + "\n" + header + "\n" + _SEED
    raw = fim_fn(lead, "\n")
    raw, _ = _trim_boundary_overlap(raw, lead, "\n")
    middle = _strip_fences(raw).rstrip("\n")
    generated = (_SEED + middle) if middle.strip() else ""

    # Assemble final = ORIGINAL verbatim + the generated tests. P1 holds by
    # construction: the original bytes are the prefix, unchanged.
    if generated:
        final = body + "\n" + header + "\n" + generated + "\n"
    else:
        final = text  # the model produced nothing — honestly, no new tests
    prefix_identical = final[:len(text)] == text

    parses = _check_parses(final, filename)

    before_ids = after_ids = ()
    outcomes_before = outcomes_after = {}
    ids_monotonic = green_preserved = False
    ran = runner is not None and parses is not False
    if ran:
        outcomes_before = runner(text)
        outcomes_after = runner(final)
        before_ids = tuple(outcomes_before)
        after_ids = tuple(outcomes_after)
        before_set, after_set = set(before_ids), set(after_ids)
        added = after_set - before_set
        ids_monotonic = before_set <= after_set and len(added) > 0
        green_preserved = (
            all(outcomes_after.get(i) == outcomes_before[i] for i in before_ids)
            and all(outcomes_after[i] in ("passed", "failed", "skipped") for i in added)
        )

    result = GrowTestsResult(
        text=final, generated=generated, before_ids=before_ids, after_ids=after_ids,
        outcomes_before=outcomes_before, outcomes_after=outcomes_after,
        prefix_identical=prefix_identical, parses=parses,
        ids_monotonic=ids_monotonic, green_preserved=green_preserved, ran=ran)
    return result


def pytest_runner(text: str) -> dict:
    """Real runner: write `text` to a temp file, run pytest with the BUILTIN
    --junit-xml channel (no plugin dep), parse outcomes from the XML via stdlib."""
    import os
    import subprocess
    import sys
    import tempfile
    import xml.etree.ElementTree as ET

    workdir = tempfile.mkdtemp(prefix="vibe-fim-grow-")
    test_path = os.path.join(workdir, "test_grow_subject.py")
    junit = os.path.join(workdir, "out.xml")
    with open(test_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "--junit-xml", junit,
         "-q", "-p", "no:cacheprovider"],
        capture_output=True, cwd=workdir, timeout=300)
    outcomes: dict = {}
    if os.path.exists(junit):
        # The XML is pytest's OWN junit output for a file we just wrote — a trusted,
        # self-generated artifact, not attacker input (running the tests already
        # executes that code). Stdlib ET is fine here; no dependency added.
        for tc in ET.parse(junit).iter("testcase"):
            name = tc.get("name") or ""
            cls = (tc.get("classname") or "").split(".")
            # junit classname is "<module>" for top-level tests, "<module>.<Class>"
            # for methods; key by function (and class, if any) within this one file.
            node_id = name if len(cls) <= 1 else f"{cls[-1]}::{name}"
            if tc.find("failure") is not None:
                outcomes[node_id] = "failed"
            elif tc.find("error") is not None:
                outcomes[node_id] = "error"
            elif tc.find("skipped") is not None:
                outcomes[node_id] = "skipped"
            else:
                outcomes[node_id] = "passed"
    return outcomes
