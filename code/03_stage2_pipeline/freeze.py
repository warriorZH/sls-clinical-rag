#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结非 O 条件的答案：记录每个答案文件的 SHA256。之后才允许读取由金标导出的 O1–O4 证据清单。"""
import argparse
import sys

from common import RUNS, Log, load_config, mark_done, now, read_json, sha256_file, write_json
import conditions as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='main,ksens,palt')
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    cfg = load_config()
    log = Log(RUNS / 'logs' / 'freeze.log')
    allc = C.conditions(cfg)
    files, missing = {}, []
    for s in a.sets.split(','):
        for c in allc[s]:
            p = RUNS / 'gen' / c['target'] / c['gen'] / '{}.jsonl'.format(c['id'])
            if p.with_suffix('.done.json').exists():
                files[str(p.relative_to(RUNS))] = sha256_file(p)
            elif not c.get('optional'):
                missing.append(c['id'])
    if missing:
        log('以下必需条件还没有完整答案：{}'.format(missing))
        sys.exit(1)
    prev = read_json(RUNS / 'freeze_nonoracle.json')
    if prev and prev.get('files') != files:
        log('注意：与上一次冻结记录不同，旧记录另存')
        write_json(RUNS / 'freeze_nonoracle.prev_{}.json'.format(now().replace(':', '')), prev)
    write_json(RUNS / 'freeze_nonoracle.json', {'time': now(), 'n_files': len(files), 'files': files})
    log('已冻结 {} 个答案文件'.format(len(files)))
    mark_done(a.job, {'n_files': len(files)})


if __name__ == '__main__':
    main()
