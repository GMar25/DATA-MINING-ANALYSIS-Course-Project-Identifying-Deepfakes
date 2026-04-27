import json, sys

def inspect(path, max_cells=40):
    nb = json.load(open(path, 'r', encoding='utf-8'))
    cells = nb['cells']
    print(f"\n{'='*60}")
    print(f"FILE: {path}")
    print(f"Total cells: {len(cells)}")
    print(f"{'='*60}")
    
    md_count = sum(1 for c in cells if c['cell_type'] == 'markdown')
    code_count = sum(1 for c in cells if c['cell_type'] == 'code')
    print(f"Markdown cells: {md_count}, Code cells: {code_count}")
    
    for i, c in enumerate(cells[:max_cells]):
        ct = c['cell_type']
        src = ''.join(c['source'])
        has_output = bool(c.get('outputs'))
        print(f"\n--- Cell {i} [{ct}] {'(HAS OUTPUT)' if has_output else ''} ---")
        print(src[:300])

for path in sys.argv[1:]:
    try:
        inspect(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
