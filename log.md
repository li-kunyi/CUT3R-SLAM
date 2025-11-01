# 251023
- check the code logic of the loop closure part
- geometry evaluation code

> scannet_test_1023_0: checked and fixed a bug in loop closure
> scannet_test_1023_1: optimization: pose lr=0
> scannet_test_1024_0: add pose refinement after gaussian update (after lc): improved
> scannet_test_1024_2: pose refinement -> GBA -> pose refinement -> GBA, densify_grad_threshold: 0.0005
> scannet_test_1024_3: pose refinement -> GBA -> pose refinement -> GBA -> pose refinement -> GBA -> pose refinement -> GBA, densify_grad_threshold: 0.0005
> scannet_test_1024_4: lc, gauissian pose refinement -> update pointmap, pose refinement -> GBA -> pose refinement -> GBA -> pose refinement -> GBA -> pose refinement -> GBA, densify_grad_threshold: 0.0005
> scannet_test_1024_5: lc, gauissian pose refinement -> update pointmap, pose refinement -> GBA -> pose refinement -> GBA, densify_grad_threshold: 0.0005, lc lr=0.005， depth=scale_depeh
> scannet_test_1025_0: lc, gauissian pose refinement -> update pointmap, pose refinement -> GBA -> pose refinement -> GBA, densify_grad_threshold: 0.0005, lc lr=0.005