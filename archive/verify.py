import json
nb = json.load(open('main_notebook.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
em_count = 0
for i, c in enumerate(cells):
    ct = c['cell_type']
    src = ''.join(c['source'])
    first_line = src.strip().split('\n')[0][:100]
    print(f'Cell {i:2d} [{ct:8s}] {first_line}')
    if '\u2014' in src:
        em_count += 1

print(f'\nTotal: {len(cells)} cells')
print(f'Cells with em dashes: {em_count}')
