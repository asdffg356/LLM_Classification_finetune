from pathlib import Path
import csv, math

OUT=Path('outputs/figures'); OUT.mkdir(parents=True, exist_ok=True)

def svg(path, w, h, body):
    Path(path).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/>{body}</svg>\n')

def line_chart():
    stages=[('Initial TF-IDF LR',1.139320,None),('Tuned TF-IDF LR',1.080877,1.07737),('DistilBERT Best Debug',1.078755,None),('TF-IDF + A/B Swap',1.078106,1.07454),('Tuned Word+Char Ensemble',1.077990,1.07430),('Pairwise TF-IDF\nDifference Model',1.017894,1.02159)]
    w,h=1400,800; L,R,T,B=110,70,90,170
    vals=[s[1] for s in stages]; ymin,ymax=1.01,1.15
    def x(i): return L+i*(w-L-R)/(len(stages)-1)
    def y(v): return h-B-(v-ymin)/(ymax-ymin)*(h-T-B)
    parts=[f'<text x="{w/2}" y="42" font-size="30" text-anchor="middle" font-family="Arial" font-weight="700">Model Evolution of the Project</text>']
    # axes/grid
    for t in [1.02,1.04,1.06,1.08,1.10,1.12,1.14]:
        yy=y(t); parts.append(f'<line x1="{L}" y1="{yy}" x2="{w-R}" y2="{yy}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{L-15}" y="{yy+5}" font-size="16" text-anchor="end" font-family="Arial">{t:.2f}</text>')
    pts=' '.join(f'{x(i):.1f},{y(v):.1f}' for i,(_,v,_) in enumerate(stages))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="4"/>')
    for i,(name,v,ps) in enumerate(stages):
        xx,yy=x(i),y(v); final=i==5
        parts.append(f'<circle cx="{xx}" cy="{yy}" r="{10 if final else 7}" fill="{"#dc2626" if final else "#2563eb"}"/>')
        parts.append(f'<text x="{xx}" y="{yy-18}" font-size="17" text-anchor="middle" font-family="Arial" font-weight="700">{v:.6f}</text>')
        if ps is not None:
            parts.append(f'<text x="{xx}" y="{yy+34}" font-size="14" text-anchor="middle" font-family="Arial" fill="#6b7280">Public {ps:.5f}</text>')
        for j,line in enumerate(name.split('\\n') if '\\n' in name else [name]):
            parts.append(f'<text x="{xx}" y="{h-B+45+20*j}" font-size="15" text-anchor="middle" font-family="Arial">{line}</text>')
    parts.append(f'<text x="22" y="{h/2}" transform="rotate(-90 22 {h/2})" font-size="20" text-anchor="middle" font-family="Arial">Validation Log Loss</text>')
    parts.append(f'<text x="{x(5)}" y="{y(1.017894)-50}" font-size="18" text-anchor="middle" font-family="Arial" fill="#dc2626" font-weight="700">Final best model</text>')
    svg(OUT/'experiment_evolution_6_stage.png', w,h,''.join(parts))

def table():
    rows=[('Initial TF-IDF LR','1.139320','0.380219','0.380760',''),('Tuned TF-IDF LR','1.080877','0.385264','0.383576','1.07737'),('DistilBERT Best Debug','1.078755','0.392222','0.379731',''),('DistilBERT Medium','1.079490','0.386991','0.371759',''),('TF-IDF + A/B Swap','1.078106','0.395442','0.396300','1.07454'),('Word+Char TF-IDF','1.078087','0.394311','0.395032',''),('Tuned Word+Char Ensemble','1.077990','0.396399','0.397240','1.07430'),('Pairwise TF-IDF Difference Model','1.017894','0.488083','0.482842','1.02159')]
    w,h=1500,650; x0,y0=40,90; rh=58; cols=[430,220,220,220,230]
    headers=['Experiment','Valid Log Loss','Valid Accuracy','Valid Macro F1','Kaggle Public Score']
    parts=[f'<text x="{w/2}" y="45" font-size="30" text-anchor="middle" font-family="Arial" font-weight="700">Final Experiment Results Summary</text>']
    x=x0
    for c,head in zip(cols,headers):
        parts.append(f'<rect x="{x}" y="{y0}" width="{c}" height="{rh}" fill="#1f2937"/>')
        parts.append(f'<text x="{x+12}" y="{y0+36}" font-size="18" font-family="Arial" fill="white" font-weight="700">{head}</text>'); x+=c
    for r,row in enumerate(rows):
        y=y0+rh*(r+1); hi='Pairwise' in row[0]; fill='#fee2e2' if hi else ('#f9fafb' if r%2 else 'white')
        x=x0
        for c,val in zip(cols,row):
            parts.append(f'<rect x="{x}" y="{y}" width="{c}" height="{rh}" fill="{fill}" stroke="#d1d5db"/>')
            parts.append(f'<text x="{x+12}" y="{y+36}" font-size="18" font-family="Arial" font-weight="{700 if hi else 400}">{val}</text>'); x+=c
    svg(OUT/'final_results_summary_table.png', w,h,''.join(parts))

def sensitivity():
    files=[('outputs/logs/pairwise_tfidf_c_tuning_results.csv','Pairwise TF-IDF C Sensitivity','C','best C=0.1'),('outputs/logs/pairwise_temperature_results.csv','Pairwise Probability Temperature Calibration','temperature','best T=1.0'),('outputs/logs/pairwise_final_ensemble_results.csv','Pairwise Ensemble Weight Sensitivity','pairwise_weight','best weight=1.0')]
    w,h=1800,560; parts=[]
    for idx,(fp,title,xcol,best) in enumerate(files):
        with open(fp, newline='', encoding='utf-8-sig') as f: data=list(csv.DictReader(f))
        xs=[float(r[xcol]) for r in data]; ys=[float(r['valid_log_loss']) for r in data]
        ox=idx*600+60; oy=80; pw=500; ph=330; ymin=min(ys)-0.002; ymax=max(ys)+0.002
        def X(v): return ox+(v-min(xs))/(max(xs)-min(xs) or 1)*pw
        def Y(v): return oy+ph-(v-ymin)/(ymax-ymin)*ph
        parts.append(f'<text x="{ox+pw/2}" y="40" font-size="22" text-anchor="middle" font-family="Arial" font-weight="700">{title}</text>')
        parts.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="white" stroke="#374151"/>')
        pts=' '.join(f'{X(x):.1f},{Y(y):.1f}' for x,y in sorted(zip(xs,ys)))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="3"/>')
        for x,yv in zip(xs,ys):
            good=abs(yv-1.017894)<1e-4
            parts.append(f'<circle cx="{X(x)}" cy="{Y(yv)}" r="{7 if good else 5}" fill="{"#dc2626" if good else "#2563eb"}"/>')
        parts.append(f'<text x="{ox+pw/2}" y="{oy+ph+45}" font-size="17" text-anchor="middle" font-family="Arial">{xcol}</text>')
        parts.append(f'<text x="{ox+pw/2}" y="{oy+ph+75}" font-size="16" text-anchor="middle" font-family="Arial" fill="#dc2626" font-weight="700">{best}; best log loss=1.017894</text>')
    svg(OUT/'pairwise_parameter_sensitivity_combined.png', w,h,''.join(parts))

if __name__=='__main__':
    line_chart(); table(); sensitivity()
    print('Generated 3 figure files in outputs/figures')
