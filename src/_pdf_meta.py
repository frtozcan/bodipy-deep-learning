import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfReader

folder = r'E:\Tox21_MWP\Makaleler'
for f in sorted(glob.glob(os.path.join(folder, '*.pdf'))):
    name = os.path.basename(f)
    print('\n' + '=' * 90)
    print('DOSYA:', name)
    print('-' * 90)
    try:
        r = PdfReader(f)
        meta = r.metadata or {}
        if meta.get('/Title'):
            print('[meta baslik]', meta.get('/Title'))
        # ilk sayfadan ilk ~35 satir (baslik/yazar/dergi/DOI genelde burada)
        txt = (r.pages[0].extract_text() or '')
        lines = [l.strip() for l in txt.split('\n') if l.strip()]
        for l in lines[:35]:
            print(l)
    except Exception as e:
        print('  HATA:', e)
