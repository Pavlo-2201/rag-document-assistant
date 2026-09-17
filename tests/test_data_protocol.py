import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from rag.llm import weight_hashes
from rag.qa import QAItem, Evidence, load_qa
from test_retrieval import make_chunks

ROOT = Path(__file__).resolve().parents[1]


def test_training_rejects_eval_pages_before_loading_a_model():
    function = runpy.run_path(str(ROOT / 'scripts/train_e5.py'))['training_pairs']
    chunks = make_chunks()
    items = [QAItem(f't-{n}', 'train', 'бытовой вопрос', True,
                    evidence=[Evidence('slug', c.doc_id, c.text[:25], c.start, c.start+25)])
             for n,c in enumerate(list({c.doc_id: c for c in chunks}.values())[:2])]
    pairs = function(items, chunks, set())
    assert len(pairs) == 2
    assert all(p['anchor'].startswith('query: ') and p['positive'].startswith('passage: ') for p in pairs)
    with pytest.raises(ValueError, match='eval-страница'):
        function(items, chunks, {items[0].evidence[0].doc_id})


def test_second_gguf_shard_changes_weight_identity(tmp_path):
    first=tmp_path/'model-00001-of-00002.gguf'
    second=tmp_path/'model-00002-of-00002.gguf'
    first.write_bytes(b'first'); second.write_bytes(b'second')
    before=weight_hashes(first)
    second.write_bytes(b'updated')
    assert weight_hashes(first) != before


def test_v2_has_balanced_splits_unique_pages_and_no_exact_old_questions():
    items=load_qa(ROOT/'data/eval/qa-v2.jsonl')
    old=load_qa(ROOT/'data/eval/qa.jsonl')
    assert len(items)==200 and len({i.id for i in items})==200
    normalize=lambda q:' '.join(q.casefold().replace('ё','е').split()).rstrip('?.!')
    assert not {normalize(i.question) for i in items} & {normalize(i.question) for i in old}
    pages=[e.doc_id for i in items for e in i.evidence]
    assert len(pages)==len(set(pages))==160
    for split in ('dev_v2','test_v2'):
        subset=[i for i in items if i.split==split]
        assert len(subset)==100 and sum(i.answerable for i in subset)==80
    provenance=json.loads((ROOT/'data/eval/qa-v2.provenance.json').read_text(encoding='utf-8'))
    assert provenance['reviewer_type']=='assistant' and provenance['human_review_status']=='pending'


def test_v2_can_be_rebuilt_byte_for_byte_on_any_platform(tmp_path):
    output = tmp_path / 'rebuilt.jsonl'
    subprocess.run([sys.executable, 'scripts/assemble_eval_v2.py', '--output', str(output)],
                   cwd=ROOT, check=True)
    assert output.read_bytes() == (ROOT / 'data/eval/qa-v2.jsonl').read_bytes()
