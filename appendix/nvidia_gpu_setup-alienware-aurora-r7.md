* Draft: 2021-01-25 (Mon)
# `Alienware-Aurora-R7`에 NVIDIA CUDA Toolkit을 설치한 기록


`$`는 `k8smaster@k8smaster-Alienware-Aurora-R7:~$`의 줄인 표현입니다.
```bash
$ nvidia-smi
Mon Jan 25 13:55:03 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    On   | 00000000:01:00.0  On |                  N/A |
| 29%   44C    P8     9W / 180W |    226MiB /  8118MiB |     13%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 1080    On   | 00000000:02:00.0 Off |                  N/A |
| 28%   29C    P8     5W / 180W |      2MiB /  8119MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1224      G   /usr/lib/xorg/Xorg                124MiB |
|    0   N/A  N/A      1352      G   /usr/bin/gnome-shell               98MiB |
+-----------------------------------------------------------------------------+
$
