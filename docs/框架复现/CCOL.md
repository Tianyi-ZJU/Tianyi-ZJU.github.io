# CCOL

https://github.com/YapengTian/CCOL-CVPR21

## co-sep

比较要命的是他用的这个[co-sep](https://github.com/rhgao/co-separation)项目，用的是比较古早的[faster-cnn-pytorch](https://github.com/jwyang/faster-rcnn.pytorch)0.4.0实现。然而这个pytorch版本太老了，和cuda无法配套。实际上还有个[pytorch1.0](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)版本，这个勉强是可以跑的。但麻烦的是co-sep这个框架用的是0.4.0的老版本，所以我们要改很多东西。

我们要用的只有co-sep库里的 [`getDetectionResults.py`](https://github.com/rhgao/co-separation/blob/master/getDetectionResults.py) 文件，还需要在README中以谷歌网盘链接形式给出的[预训练模型](https://drive.google.com/file/d/1fiS3uBiZSsPkKfxr0mo9IIOhSOirApLH/view?usp=drive_link)，然后把主战场切换到faster-cnn-pytorch。

??? tip "Environment"
	environment : python3.10，pytorch1.13.0+cuda11.7

先clone下来：`git clone -b pytorch-1.0 https://github.com/jwyang/faster-rcnn.pytorch.git`

我们首先参考这篇[文章](https://blog.csdn.net/ljtlyk/article/details/129945955)。从4.1看到4.3即可。不要尝试运行任何的 `make.sh` 文件。

一个可能需要改动的地方是在`lib/model/utils/config.py` 的292行（可能不是这一行，注意辨析）改成 `__C.ANCHOR_SCALES = [4,8,16,32]` 。我不确定如果不改会不会有问题（反正我改了）。

其中，在4.3.1编译coco api的时候可能会有如下报错：

```
Error compiling Cython file:
------------------------------------------------------------
...
	cdef np.ndarray[np.double_t, ndim=1] np_poly
	n = len(poly)
	Rs = RLEs(n)
	for i, p in enumerate(poly):
		np_poly = np.array(p, dtype=np.double, order='F')
		rleFrPoly( <RLE*>&Rs._R[i], <const double*> np_poly.data, len(np_poly)/2, h, w )
																			  ^
------------------------------------------------------------

_mask.pyx:258:78: Cannot assign type 'double' to 'siz' (alias of 'unsigned long')
```

把 `_mask.pyx` 第258行改成 
```
rleFrPoly( <RLE*>&Rs._R[i], <const double*> np_poly.data, <unsigned long>(len(np_poly)/2), h, w )
```
即可。

之后我们来魔改他的 `getDetectionResults.py`· 。这个程序如果直接跑会有很多问题，因为它是根据老版本的pytorch0.4 的demo 来写的，我们需要对照新版本的demo来修改它。

以下是**必须**要修改的地方：

1. 默认使用的res101务必不要去动他（我在这吃了很多苦头）。一些路径参数嫌麻烦可以直接在args-default里面改。

2. `load_name` 是加载预训练模型的路径，可以直接把这个改成我们模型的路径。

3. 原来的导入的 `imread` 函数已弃用，可以改成 `from imageio.v2 import imread`

4. 原来导入 `nms` 的那一行改成 `from model.roi_layers import nms`

5. 找到与下面这一段类似的代码直接修改：
    ```python
    if vis:
      im2show = np.copy(im)
    for j in xrange(1, len(pascal_classes)):
      inds = torch.nonzero(scores[:,j]>thresh).view(-1)
      # if there is det
      if inds.numel() > 0:
        cls_scores = scores[:,j][inds]
        _, order = torch.sort(cls_scores, 0, True)
        if args.class_agnostic:
          cls_boxes = pred_boxes[inds, :]
        else:
          cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
    
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        cls_dets = cls_dets[order]
        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
        cls_dets = cls_dets[keep.view(-1).long()]
    
        #get statistics for all boxes: frame_index, class_label, confidence_score, four cordinates
        boxCount = cls_dets.shape[0]
        box_array = np.empty([boxCount,7])
        for b_index in range(boxCount):
          box_array[b_index][0] = int(imglist[num_images][:-4])
          box_array[b_index][1] = j
          box_array[b_index][2] = cls_dets[b_index,-1]
          box_array[b_index][3:] = cls_dets[b_index, :-1].cpu().numpy()
        if count == 0:
          array2save = box_array
          count = count + boxCount
        else:
          array2save = np.concatenate((array2save, box_array), axis=0)
        if vis:
          im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)
    ```

以下是**可能**需要修改的地方：

1. `fasterRCNN.load_state_dict(checkpoint['model'],strict=False)` 加载模型的地方添加 `strict=False`
2. 最后的保存npy那一块代码可能缩进有问题，需要缩进一次。

可能还有改动，但是我忘了，但应该没有大问题了，还有问题可以根据报错自己解决一下。

至此这个模型终于能跑了。

## CCOL







