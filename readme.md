* [Fractal2DCategorySearcher](#Fractal2DCategorySearcher)  
  カテゴリ探索プログラム
* [Fractal2DCategorySearcherGrid](#Fractal2DCategorySearcherGrid)  
  グリッドサーチ版カテゴリ探索プログラム
* [PyFractal2DRenderer](#PyFractal2DRenderer)
  フラクタル描画Pythonモジュール
* [Fractal2DGenData](#Fractal2DGenData)
  PyTorchデータローダサンプル
* [Fractal2DRenderer_cpu](#Fractal2DRenderer_cpu)
  フラクタル描画C++ライブラリ

---

# Fractal2DCategorySearcher

有効なカテゴリを探索します．

## Install

### Requirements

* gcc (C++14対応)
* cmake
* OpenCV 2+

### Compile

```sh
mkdir build
cd build
cmake .. -DOpenCV_DIR=[OpenCVConfig.cmakeがあるディレクトリ]
make Fractal2DCategorySearcher
```

## How to Use

先に出力先ディレクトリを作成する必要があります．

```sh
mkdir c1kr03
./Fractal2DCategorySearcher --fn_prefix=./c1kr03/cat_ --categories=1000
```

```text
options:
    --fn_prefix
        output file prefix including path (mandatory)
    --iters
        number of parameter search iteration (either mandatory)
    --categories
        number of valid category count to stop iteration (either mandatory)
    -h, --help
        show this message
    --paramgen_seed
        random seed for parameter generation
    --use_checkpoint
        checkpoint file to use
    --pointgen_seed
        random seed for point generation
    --width
        internal image width
    --height
        internal image height
    --npts
        internal number of points to generate
    --thresh
        threshold for validation of occupancy on a rendered image
    --checkpoint_iters
        output duration of checkpoint for paramgen random state (0 to disable)
    --range_nmaps_min
        lower range of number of maps
    --range_nmaps_max
        upper range of number of maps
    --range_param_min
        lower range of parameter
    --range_param_max
        upper range of parameter
    --enable_image_output
        flag for category image output
    --debug
        debug flag
```

---

# Fractal2DCategorySearcherGrid

グリッドで有効なカテゴリを探索します．
重複パターンは探索しません．
引数に応じてランダム摂動も入れられます．

## Install

### Requirements

* gcc (C++14対応)
* cmake
* OpenCV 2+

### Compile

```sh
mkdir build
cd build
cmake .. -DOpenCV_DIR=[OpenCVConfig.cmakeがあるディレクトリ]
make Fractal2DCategorySearcher
```

## How to Use

先に出力先ディレクトリを作成する必要があります．

```sh
mkdir c1kr03
./Fractal2DCategorySearcherGrid --fn_prefix=./c1kr03/cat_ --grid_span=0.01 --grid_perturbate_range=0.0005 --grid_perturbate_type=uniform
```

```text
options:
    --fn_prefix
        output file prefix including path (mandatory)
    --grid_span
        grid span (mandatory)
    --grid_perturbate_range
        perturbation range of grid (mandatory)
    --grid_perturbate_type
        perturbation random distribution type (uniform, normal) (mandatory)
    -h, --help
        show this message
    --paramgen_seed
        random seed for parameter generation
    --use_checkpoint
        checkpoint file to use
    --pointgen_seed
        random seed for point generation
    --width
        internal image width
    --height
        internal image height
    --npts
        internal number of points to generate
    --thresh
        threshold for validation of occupancy on a rendered image
    --checkpoint_iters
        output duration of checkpoint for paramgen random state (0 to disable)
    --range_nmaps_min
        lower range of number of maps
    --range_nmaps_max
        upper range of number of maps
    --range_param_min
        lower range of parameter
    --range_param_max
        upper range of parameter
    --p_by_det
        using determinant of parameter for probability
    --enable_image_output
        flag for category image output
```

---

# PyFractal2DRenderer

フラクタルを描画するPythonモジュールです．
この関数を使うDatasetを作ってDataLoaderに組み込むことができます．

## Install

### Requirements

* Python (3.8で確認)
* PyTorch (x.xで確認)
* OpenCV 4+

### Compile

```sh
# 無くてもビルド可能
export OpenCV_ROOT=[OpenCVのincludeディレクトリのあるディレクトリ]
# 無くてもビルド可能
export flann_ROOT=[flannのincludeディレクトリのあるディレクトリ]

python setup.py build
```

## How to use

```python
import sys
sys.path.insert(1, "/path/to/dynamic_lib")
import numpy as np
import torch #required to import former!!
import PyFractal2DRenderer as fr

maps_shida=[
    [0,0,0,0.16,0,0,0.01],
    [0.85,0.04,-0.04,0.85,0,1.60,0.85],
    [0.20,-0.26,0.23,0.22,0,1.60,0.07],
    [-0.15,0.28,0.26,0.24,0,0.44,0.07]
]

mapss=[
    maps_shida
]

npts=100000
pointgen_seed=100 # 変換選択に用いる乱数シード
pts = fr.generate(npts, mapss, pointgen_seed) #np.array [N,npts,2]

width=256
height=256
patch_mode=0 # 0のみ対応
flip_flg=0 # 0:フリップ無し 1: 左右フリップ 2: 上下フリップ 3: 上下左右フリップ
patchgen_seed=100 # パッチ生成に用いる乱数シード
imgs = fr.render(pts, width, height, patch_mode, flip_flg, patchgen_seed) #torch.Tensor(ByteTensor) [N,H,W,C]
```

---

# Fractal2DGenData

Datasetのサンプル

## Install

PyFractal2DRendererへのPythonモジュールパスを通しておいてください．

```sh
export PYTHONPATH="/path/to/binary/of/PyFractal2DRenderer:${PYTHONPATH}"
```

## How to use

```python
import numpy as np
import torch,torchvision
from Fractal2DGenData import Fractal2DGenData

def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    seed = info.dataset.patchgen_seed + worker_id
    info.dataset.patchgen_rng = np.random.default_rng(seed)

transform=torchvision.transforms.Compose([
   torchvision.transforms.ToPILImage(),
   torchvision.transforms.RandomCrop([224,224]),
   torchvision.transforms.ToTensor(),
   torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
dataset = Fractal2DGenData(
    param_dir="/path/to/collection/of/00000.csv/../..", width = 362, height = 362, npts: = 100000, patch_mode:int = 0,
    patch_num:int = 10, patchgen_seed:int = 100, pointgen_seed:int = 100, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=8,
                    shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)

for i,(d,l) in enumerate(loader):
    # DO SOMETHING

```

---

# Fractal2DRenderer_cpu

フラクタルを描画するC++ライブラリです．(CPU版)
(以下，超雑な説明)

## Install

```sh
mkdir build
cd build
cmake .. -DOpenCV_DIR=<> -DTorch_DIR=<> -Dflann_DIR=<>
make Fractal2DRenderer_cpu_test
```

## How to use

```cpp
#include<Fractal2DRenderer_cpu.h>
int main(){
    const std::vector<IFSMap> maps_shida={
        {0,0,0,0.16,0,0,0.01},
        {0.85,0.04,-0.04,0.85,0,1.60,0.85},
        {0.20,-0.26,0.23,0.22,0,1.60,0.07},
        {-0.15,0.28,0.26,0.24,0,0.44,0.07}
    };
    const std::vector<std::vector<IFSMap>> mapss={maps_shida};
    const int ninstances=mapss.size();
    const int npts = 100000;
    const int pointgen_seed = 100;
    std::vector<double> pts(ninstances*npts*2, 0);
    generatePoints_cpu(pts.data(), npts, mapss, 100);

    const int width=256;
    const int height=256;
    const int patch_mode=0;
    const int flip_flg=0;
    const int patchgen_seed=0;
    std::vector<unsigned int> imgs(ninstances*height*width*3, 0);
    renderPoints_cpu(pts.data(), npts, ninstances, imgs.data(), width, height, patch_mode, flip_flg, patchgen_seed);

    {
        FILE *fp=fopen("pts.bin","wb");
        fwrite(pts.data(),sizeof(double)*pts.size(),fp);
        fclose(fp);
    }
    {
        FILE *fp=fopen("imgs.bin","wb");
        fwrite(imgs.data(),sizeof(unsigned int)*imgs.size(),fp);
        fclose(fp);
    }
    return 0;
}
```
