"""Summarize completed Qwen3.5 benchmark JSON files without mixing smoke runs."""
import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    rows = []
    totals = {k: 0 for k in ('baseline_tokens', 'flashmtp_tokens', 'baseline_seconds', 'flashmtp_seconds', 'accepted_tokens', 'verification_steps', 'turns', 'greedy_equal_turns')}
    for path in sorted(root.glob('*.json')):
        if path.name == 'summary.json':
            continue
        data = json.loads(path.read_text())
        if 'overall' not in data or 'records' not in data:
            continue
        args = data['args']
        stats = data['overall']
        b_tokens = sum(r['baseline']['num_tokens_for_decode_rate'] for r in data['records'])
        s_tokens = sum(r['flashmtp']['num_tokens_for_decode_rate'] for r in data['records'])
        b_seconds = sum(r['baseline']['decode_wall_time'] for r in data['records'])
        s_seconds = sum(r['flashmtp']['decode_wall_time'] for r in data['records'])
        acceptance = [n for r in data['records'] for n in r['flashmtp']['acceptance_lengths']]
        assert stats['num_turns'] == len(data['records']) > 0
        assert b_seconds > 0 and s_seconds > 0
        assert abs(stats['token_weighted_speedup'] - (b_seconds / b_tokens) / (s_seconds / s_tokens)) < 1e-7
        row = {
            'dataset': args['dataset'], 'requested_samples': args['max_samples'],
            'turns': stats['num_turns'], 'mean_acceptance_length': stats['avg_accept_length'],
            'token_weighted_speedup': stats['token_weighted_speedup'],
            'baseline_tokens_per_second': b_tokens / b_seconds,
            'flashmtp_tokens_per_second': s_tokens / s_seconds,
            'greedy_equal_turns': data['greedy_equal_turns'],
            'source': str(path),
        }
        rows.append(row)
        for key, value in [('baseline_tokens', b_tokens), ('flashmtp_tokens', s_tokens), ('baseline_seconds', b_seconds), ('flashmtp_seconds', s_seconds), ('accepted_tokens', sum(acceptance)), ('verification_steps', len(acceptance)), ('turns', stats['num_turns']), ('greedy_equal_turns', data['greedy_equal_turns'])]:
            totals[key] += value
    payload = {'completed_datasets': len(rows), 'rows': rows, 'totals': totals}
    if rows:
        payload['combined_token_weighted_speedup'] = (totals['baseline_seconds'] / totals['baseline_tokens']) / (totals['flashmtp_seconds'] / totals['flashmtp_tokens'])
        payload['combined_mean_acceptance_length'] = totals['accepted_tokens'] / totals['verification_steps']
        (root / 'summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        with (root / 'summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2, ensure_ascii=False))
