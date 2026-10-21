'''
Print seed texts and the new generated inputs from a gen_diff.py run.

usage:
  python print_inputs.py
  python print_inputs.py ./generated_inputs/synonym
  python print_inputs.py ./generated_inputs/synonym --generated-only
'''

import argparse
import glob
import json
import os

from configs import ID2LABEL, bcolors


def load_runs(paths):
    files = sorted({f for p in paths for f in glob.glob(os.path.join(p, '**', 'results.json'), recursive=True)})
    runs = []
    for f in files:
        with open(f, encoding='utf-8') as fh:
            run = json.load(fh)
        run['dir'] = os.path.dirname(f)
        runs.append(run)
    return runs


def names(preds):
    return [ID2LABEL[int(p)] for p in preds]


def print_record(r, generated_only=False):
    if generated_only and r['status'] != 'generated':
        return False
    if r['status'] not in ('generated', 'already_differ'):
        return False
    print(bcolors.WARNING + '-' * 72 + bcolors.ENDC)
    print(bcolors.BOLD + f"seed {r['seed_idx']} [{r['status']}]" + bcolors.ENDC)
    print(f"  SEED text : {r['orig_text']}")
    print(f"  SEED preds: {names(r['orig_predictions'])}  {r['orig_predictions']}")
    gen = r.get('generated_text') or r['orig_text']
    print(bcolors.OKGREEN + f"  NEW  text : {gen}" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"  NEW  preds: {names(r['predictions'])}  {r['predictions']}" + bcolors.ENDC)
    if r.get('iterations') is not None:
        print(f"  iterations: {r['iterations']}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Print DeepXplore seed vs generated ticket texts')
    parser.add_argument('paths', nargs='*', default=['./generated_inputs'],
                        help='directories containing results.json')
    parser.add_argument('--generated-only', action='store_true',
                        help='skip seeds that already disagreed without editing')
    args = parser.parse_args()

    runs = load_runs(args.paths)
    if not runs:
        raise SystemExit(f'no results.json found under {args.paths}')

    total = 0
    for run in runs:
        print(bcolors.HEADER + f"\n=== {run['args']['transformation']} ({run['dir']}) ===" + bcolors.ENDC)
        n = 0
        for r in run['records']:
            if print_record(r, generated_only=args.generated_only):
                n += 1
                total += 1
        if n == 0:
            print('(none)')
    print(bcolors.HEADER + f'\nTotal printed: {total}' + bcolors.ENDC)


if __name__ == '__main__':
    main()
