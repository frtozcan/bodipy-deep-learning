"""ref_scan.py - Makaleler/ klasorundeki PDF'lerin ilk sayfasindan baslik/ozet cikar."""
import os, sys, fitz
sys.stdout.reconfigure(encoding='utf-8')
D = r'E:\Tox21_MWP\Makaleler'
for f in sorted(os.listdir(D)):
    if not f.lower().endswith('.pdf'):
        print('== [PDF DEGIL]', f); continue
    try:
        doc = fitz.open(os.path.join(D, f))
        t = doc[0].get_text().replace('\n', ' ')
        t = ' '.join(t.split())
        print('\n' + '=' * 90)
        print('DOSYA:', f)
        print(t[:900])
    except Exception as e:
        print('\n== [HATA]', f, '->', e)
