#!/usr/bin/env python3
"""Официальные открытые веса, фиксированные ревизии и проверка LFS SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ENCODER_FILES = ['config.json', 'modules.json', 'sentence_bert_config.json', '1_Pooling/config.json',
                 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                 'sentencepiece.bpe.model', 'model.safetensors']
PROFILES = {
    'qwen4b': ('Qwen/Qwen3-4B-GGUF', 'bc640142c66e1fdd12af0bd68f40445458f3869b', 'models/qwen4b', ['Qwen3-4B-Q4_K_M.gguf']),
    'e5-base': ('intfloat/multilingual-e5-base', 'd128750597153bb5987e10b1c3493a34e5a4502a', 'models/e5-base', ENCODER_FILES),
    'e5-large': ('intfloat/multilingual-e5-large', '3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3', 'models/e5-large', ENCODER_FILES),
    'qwen3b': ('Qwen/Qwen2.5-3B-Instruct-GGUF', '7dabda4d13d513e3e842b20f0d435c732f172cbe', 'models/qwen3b', ['qwen2.5-3b-instruct-q4_k_m.gguf']),
    'judge7b': ('Qwen/Qwen2.5-7B-Instruct-GGUF', 'bb5d59e06d9551d752d08b292a50eb208b07ab1f', 'models/judge',
                ['qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf', 'qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf'])}


def file_hash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile', choices=PROFILES, nargs='+')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--manifest', default='results/quality/models.lock.json')
    args = p.parse_args()
    curl = shutil.which('curl.exe') or shutil.which('curl')
    if not curl:
        raise RuntimeError('нужен curl')
    common = [curl, '--location', '--fail', '--silent', '--show-error', '--retry', '3', '--connect-timeout', '20']
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    for name in args.profile:
        repo, revision, directory, files = PROFILES[name]
        tree = json.loads(subprocess.check_output(common + [f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true'], text=True, encoding='utf-8'))
        entries = {e['path']: e for e in tree}
        records = []
        for filename in files:
            path = Path(directory) / filename
            entry = entries[filename]
            expected = entry.get('lfs', {}).get('oid')
            valid = path.exists() and path.stat().st_size == entry['size']
            if valid and expected:
                valid = file_hash(path) == expected
            if not valid:
                if args.verify_only:
                    raise ValueError(f'нет или повреждён {path}')
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix + '.download')
                subprocess.run(common + ['--output', str(temp), f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'], check=True)
                if temp.stat().st_size != entry['size'] or (expected and file_hash(temp) != expected):
                    raise ValueError(f'контрольная сумма не совпадает: {path}')
                temp.replace(path)
            records.append({'path': str(path).replace('\\', '/'), 'bytes': path.stat().st_size,
                            'sha256': file_hash(path), 'hf_lfs_sha256': expected})
        manifest[name] = {'repository': repo, 'revision': revision, 'files': records}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(name, 'verified', flush=True)


if __name__ == '__main__':
    main()
