"""Answer grading for Web Search for Coding Agents: does the agent's final call do what the question asks?

The agent ends each question with one call (JSON: method and path, or a SOAP operation and the service address it
is sent to). It passes when that call is one of the question's right answers in answer-key.json:
- REST: same method; same path segment by segment, where a placeholder ({id}) or a version (v1) in the right
  answer matches any segment, and the agent's own placeholder matches only a placeholder; a leading /api is
  optional; inside parentheses the key names must match (values and order do not); when the right answer expands
  a name ($expand=... in its address, or require_query), the call must expand it too: the bare call returns only
  a link to the data.
- SOAP: same operation (a trailing _Request is optional) sent to the Workday service that owns it.
A question listed as unscored is graded None and left out of every count. A wrong answer also gets one failure type
(Grade.kind): no readable answer, no call, or, from the right answer it came nearest to, the first check it failed.
grade_call grades against a given list of right answers; the harness passes the list from each question row.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path

GRADER_VERSION = 'answer-grader-v1'
KEY_FILE = Path(__file__).resolve().parent / 'answer-key.json'
# SAP prints some form-section keys both as sectIndex and sectionIndex.
LABEL_ALIAS = {'sectindex': 'sectionindex'}

UNREADABLE, NO_CALL = 'unreadable answer', 'no call'
WRONG_METHOD, WRONG_SOAP, WRONG_PATH, WRONG_KEYS, NO_EXPAND = (
    'wrong method', 'wrong SOAP service/operation', 'wrong path', 'wrong key labels', 'missing expand')
# How far a miss got through the checks, least to most: the answer is named by the right answer it came nearest to.
NEARNESS = (WRONG_METHOD, WRONG_SOAP, WRONG_PATH, WRONG_KEYS, NO_EXPAND)


@dataclass(frozen=True)
class Grade:
    passed: bool | None
    reason: str
    kind: str | None = None


def _keys(inside: str) -> str:
    labels = (x.split('=')[0].strip().lower() for x in inside.split(',') if '=' in x)
    return '(' + ','.join(sorted(LABEL_ALIAS.get(l, l) for l in labels)) + ')'


def _segments(path: str) -> list[str]:
    path = re.sub(r'^https?://[^/]+', '', (path or '').split('?')[0]).rstrip('/')
    out = []
    for s in path.split('/'):
        s = re.sub(r'\(([^)]*)\)', lambda m: _keys(m.group(1)), s)
        free = re.fullmatch(r'\{[^}]*\}|v\d+(\.\d+)?', s)
        if s:
            out.append('{}' if free else s.lower())
    return out[1:] if out[:1] == ['api'] else out


def _service(address: str) -> str:
    m = re.search(r'/ccx/service/[^/]+/([^/?]+)', address)
    assert m, f'a SOAP right answer without a Workday service: {address}'
    return m.group(1).lower()


def _expands(address: str) -> list[str]:
    """The top-level names a right answer expands ($expand=a,b/c,d(...)): the bare call returns only links to them."""
    m = re.search(r'[?&]\$expand=([^&]*)', address)
    if not m:
        return []
    names, depth, cur = [], 0, ''
    for ch in m.group(1) + ',':
        depth += (ch == '(') - (ch == ')')
        if ch == ',' and depth == 0:
            names.append(re.split(r'[(/]', cur)[0].strip())
            cur = ''
        else:
            cur += ch
    return [n for n in names if n]


def _op(s: str | None) -> str:
    return re.sub(r'_Request$', '', (s or '').strip())


def _same(got: list[str], want: list[str]) -> bool:
    # a placeholder in the right answer matches anything; the agent's own placeholder matches only a placeholder
    return len(got) == len(want) and all(a == b or b == '{}' for a, b in zip(got, want))


def _rest_miss(call: dict, right: dict) -> str | None:
    """None when the call is this right answer, else the first check it fails."""
    if (call.get('method') or '').upper() != right['method'].upper():
        return WRONG_METHOD
    got, want = _segments(call.get('path') or ''), _segments(right['address'])
    if not _same(got, want):
        bare = lambda segments: [re.sub(r'\(.*\)$', '', s) for s in segments]
        return WRONG_KEYS if _same(bare(got), bare(want)) else WRONG_PATH
    query = str(call.get('path')) + '&' + '&'.join(map(str, call.get('fields') or []))
    need = ([right['require_query']] if right.get('require_query') else []) + _expands(right['address'])
    return None if all(re.search(r'(?:^|[?&\s])\$?expand=[^&\s]*\b' + re.escape(n) + r'\b', query, re.I) for n in need) else NO_EXPAND


@functools.cache
def _key() -> dict:
    return json.loads(KEY_FILE.read_text())


def grade(question_id: str, answer: str | dict | None) -> Grade:
    key = _key()
    if question_id in key['unscored']:
        return Grade(None, key['unscored'][question_id])
    entry = key['entries'].get(question_id)
    if entry is None:
        raise KeyError(f'{question_id} is not in {key["version"]}')
    return grade_call(entry['accepted'], answer)


def grade_call(accepted: list[dict], answer: str | dict | None) -> Grade:
    try:
        call = json.loads(answer) if isinstance(answer, str) else (answer or {})
    except ValueError:
        return Grade(False, 'no readable answer', UNREADABLE)
    if not isinstance(call, dict) or not (call.get('path') or call.get('operation')):
        return Grade(False, 'no call in the answer', NO_CALL)
    misses = []
    for right in accepted:
        if right.get('soap'):
            # Workday owns each operation in one service; the call must be sent to that service
            sent_to = [x.lower() for x in re.split(r'[/?]', call.get('path') or '')]
            if call.get('operation') and _op(call['operation']) == _op(right['soap']) and _service(right['address']) in sent_to:
                return Grade(True, f"SOAP {right['soap']}")
            misses.append(WRONG_SOAP)
        else:
            miss = _rest_miss(call, right)
            if miss is None:
                return Grade(True, f"{right['method']} {right['address']}")
            misses.append(miss)
    return Grade(False, 'not one of the right answers', max(misses, key=NEARNESS.index))
