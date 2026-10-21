# GitHub Release용으로 학습된 가중치와 전체 지표를 묶습니다.
# 사용법: Colab에서 GOTSF_Colab_Reproduction.ipynb를 끝까지 실행한 뒤, 이 파일 내용을 새 셀에 붙여넣고 실행하세요.
# 결과: gotsf_release/gotsf_checkpoints.zip (Release에 첨부), gotsf_release/all_metrics.csv (저장소 results/에 추가)
import glob
import os
import re
import zipfile

import numpy as np
import pandas as pd

REPO = "/content/gotsf" if os.path.isdir("/content/gotsf") else os.path.abspath("gotsf")   # 노트북이 클론한 위치
OUT = os.path.join(os.path.dirname(REPO), "gotsf_release")
os.makedirs(OUT, exist_ok=True)

# 1) 가중치: 학습이 끝난 실험(train_loss.npy가 있는 폴더)만. quick 모드 결과(checkpoints_quick/)는 제외
exps = sorted(d for d in glob.glob(os.path.join(REPO, "checkpoints", "*"))
              if os.path.exists(os.path.join(d, "train_loss.npy")))
zip_path = os.path.join(OUT, "gotsf_checkpoints.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in exps:
        for f in ("checkpoint.pth", "args.pk", "train_loss.npy"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                zf.write(p, os.path.relpath(p, REPO))   # 압축 안 경로: checkpoints/<실험>/<파일>
print(f"가중치: 실험 {len(exps)}개 → {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")

# 2) 지표: results/<실험>/(l, u)[_max|_expectation]/metrics.npy → CSV 하나 (예측 배열은 수 GB라 제외)
rows = []
for f in sorted(glob.glob(os.path.join(REPO, "results", "*", "*", "metrics.npy"))):
    exp, folder = f.split(os.sep)[-3:-1]
    m = re.fullmatch(r"\(([-\d.e]+), ([-\d.e]+)\)(?:_(max|expectation))?", folder)
    if m:
        mae, mse, rmse = (float(x) for x in np.load(f)[:3])
        rows.append(dict(experiment=exp, interval_l=float(m.group(1)), interval_u=float(m.group(2)),
                         patching=m.group(3) or "none", mae=mae, mse=mse, rmse=rmse))
csv_path = os.path.join(OUT, "all_metrics.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"지표: {len(rows)}행 → {csv_path}")

try:   # Colab이면 바로 내려받기
    from google.colab import files
    files.download(zip_path)
    files.download(csv_path)
except ImportError:
    pass
